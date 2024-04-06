import os
import csv
import pandas as pd
import math
import warnings
import mplfinance as mpf
import numpy as np
from datetime import datetime, timedelta

# ignore pandas warning
warnings.filterwarnings('ignore')


def calculate_ema(prices, days):
    """
    計算一系列價格的指數移動平均值（EMA）。
    """
    return prices.ewm(span=days, adjust=False).mean()


def simulate_trades_with_volume(data, volume, ema_days):
    """
    根據EMA和交易量模擬交易策略，返回交易和總利潤。
    """
    data[f'EMA{ema_days}'] = calculate_ema(data['Close'], ema_days)
    trades = []
    total_profit = 0
    position_open = False
    last_buy_price = None
    peak_price_after_buy = None

    for i in range(1, len(data)):
        if position_open:
            """
            短波段停利賣出
            短線最高未實現獲利回吐30% 
            """
            peak_price_after_buy = max(peak_price_after_buy, data['Close'].iloc[i])
            if (peak_price_after_buy - data['Close'].iloc[i]) / peak_price_after_buy >= 0.3:
                profit = (data['Close'].iloc[i] - last_buy_price) * volume
                total_profit += profit
                trades.append(('Sell', data['Date'].iloc[i], data['Close'].iloc[i], profit, total_profit))
                position_open = False

        if not position_open:
            """
            短波段買進
            K棒收盤價上穿23EMA並高於23EMA的 3%以上,或是排連續2支以上(包含 2支)翻多站上23EMA的1%以上
            短線停利後,如K棒又拉回到 23EMA均線上方的1%以下且次支 K棒又站上23EMA均線上方2%以上
            """
            buy_condition_1 = data['Close'].iloc[i] > data[f'EMA{ema_days}'].iloc[i] * 1.03
            buy_condition_2 = data['Close'].iloc[i] > data[f'EMA{ema_days}'].iloc[i] * 1.01 and data['Close'].iloc[
                i - 1] > data[f'EMA{ema_days}'].iloc[i - 1] * 1.01
            if buy_condition_1 or buy_condition_2:
                position_open = True
                last_buy_price = data['Close'].iloc[i]
                peak_price_after_buy = data['Close'].iloc[i]
                trades.append(('Buy', data['Date'].iloc[i], data['Close'].iloc[i], 0, total_profit))

        if position_open:
            """
            短波段賣出
            K排收盤價下穿23EMA,並低於23EMA的3%以 上,或是K棒連續2支以上(包含2支)翻空 跌破23EMA的1%以上。
            """
            sell_condition_1 = data['Close'].iloc[i] < data[f'EMA{ema_days}'].iloc[i] * 0.97
            sell_condition_2 = data['Close'].iloc[i] < data[f'EMA{ema_days}'].iloc[i] * 0.99 and data['Close'].iloc[
                i - 1] < data[f'EMA{ema_days}'].iloc[i - 1] * 0.99
            if sell_condition_1 or sell_condition_2:
                profit = (data['Close'].iloc[i] - last_buy_price) * volume
                total_profit += profit
                trades.append(('Sell', data['Date'].iloc[i], data['Close'].iloc[i], profit, total_profit))
                position_open = False

    return trades, total_profit


def summarize_trades(trades, start_date, end_date):
    """
    篩選指定起始日期和結束日期內的交易，總結利潤並計算正利潤和負利潤的次數。
    """
    # Convert start_date and end_date strings to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter trades within the specified date range
    filtered_trades = [trade for trade in trades if start_date <= pd.to_datetime(trade[1]) <= end_date]

    positive_profit_count = sum(1 for trade in filtered_trades if trade[3] > 0)
    negative_profit_count = sum(1 for trade in filtered_trades if trade[3] < 0)
    interval_profit = sum(trade[3] for trade in filtered_trades)

    print(f"從 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} 的交易：")
    print(f"正利潤交易次數： {positive_profit_count}")
    print(f"負利潤交易次數： {negative_profit_count}")
    print(f"區間利潤： {interval_profit}")
    print(f"總利潤（所有交易）： {trades[-1][-1] if trades else 0}")


def optimize_ema(data, start_date, end_date, volume, ema_range):
    """
    在指定範圍內優化EMA設置，旨在最大化區間利潤，最大化正利潤交易，並最小化負利潤交易。
    """
    # 篩選指定日期範圍內的數據
    filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

    best_settings = {
        "EMA": None,
        "Positive Profit Trades": 0,
        "Negative Profit Trades": float("inf"),
        "Interval Profit": 0
    }

    for ema_days in range(ema_range[0], ema_range[1] + 1):
        trades, _ = simulate_trades_with_volume(filtered_data, volume, ema_days)
        positive_profit_count = sum(1 for trade in trades if trade[3] > 0)
        negative_profit_count = sum(1 for trade in trades if trade[3] < 0)
        interval_profit = sum(trade[3] for trade in trades)

        # 部分 1: 更高區間利潤
        higher_profit = interval_profit > best_settings["Interval Profit"]

        # 部分 2: 等於區間利潤但有更多正收益交易
        equal_more_pos = (
                interval_profit == best_settings["Interval Profit"] and
                positive_profit_count > best_settings["Positive Profit Trades"]
        )

        # 部分 3: 等於區間利潤，等於正收益交易但負收益交易較少
        equal_pos_less_neg = (
                interval_profit == best_settings["Interval Profit"] and
                positive_profit_count == best_settings["Positive Profit Trades"] and
                negative_profit_count < best_settings["Negative Profit Trades"]
        )

        if higher_profit or equal_more_pos or equal_pos_less_neg:
            best_settings.update({
                "EMA": ema_days,
                "Positive Profit Trades": positive_profit_count,
                "Negative Profit Trades": negative_profit_count,
                "Interval Profit": interval_profit
            })

    return best_settings


def read_stock_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    return data


def write_to_csv(file_path, mode, data_rows):
    """
    處理寫入 CSV 檔案的函數。
    :param file_path: CSV 檔案的路徑。
    :param mode: 寫入模式，'w' 表示寫入（覆蓋原有內容），'a' 表示附加（在原有內容後新增）。
    :param data_rows: 要寫入的資料。一個包含多行的列表，每行是一個值列表。
    """
    with open(file_path, mode, newline='') as file:
        writer = csv.writer(file)
        if mode == 'w':
            # Write headers if we're in write mode
            writer.writerow(
                ['Stock', 'Optimal EMA', 'Positive Profit Trades', 'Negative Profit Trades', 'Interval Profit'])
        writer.writerows(data_rows)


def Backtest2Csv(volume, start_date, end_date, output_csv_path, dataPath):
    # Initialize CSV with headers
    write_to_csv(output_csv_path, 'w', [])

    stockFileList = os.listdir(dataPath)
    all_optimization_results = []
    for stock in stockFileList:
        print(stock)
        file_path = os.path.join(dataPath, stock)  # 替換成你的CSV檔案路徑
        sample_data = read_stock_data(file_path)

        ema_days = 23  # 假設的EMA天數

        # 執行模擬交易
        trades, total_profit = simulate_trades_with_volume(sample_data, volume, ema_days)

        # 篩選和總結交易
        summarize_trades(trades, start_date, end_date)

        # 優化EMA
        ema_range = (5, 70)  # 測試EMA設置的範圍
        optimal_ema_settings = optimize_ema(sample_data, start_date, end_date, volume, ema_range)

        # 第一部分：檢查 'EMA' 是否不為 None
        ema_exists = optimal_ema_settings['EMA'] is not None
        # 第二部分：檢查 'Negative Profit Trades' 是否不是無窮大
        negative_profit_not_inf = not math.isinf(optimal_ema_settings['Negative Profit Trades'])
        # 第三部分：確保不是所有的值（正利潤交易、負利潤交易、間隔利潤）都是零
        not_all_zero = not (optimal_ema_settings['Positive Profit Trades'] == 0 and optimal_ema_settings[
            'Negative Profit Trades'] == 0 and optimal_ema_settings['Interval Profit'] == 0)

        if ema_exists and negative_profit_not_inf and not_all_zero:
            all_optimization_results.append([
                stock.split('.')[0],  # 去掉文件擴展名
                optimal_ema_settings['EMA'],
                optimal_ema_settings['Positive Profit Trades'],
                optimal_ema_settings['Negative Profit Trades'],
                optimal_ema_settings['Interval Profit']
            ])
            print(optimal_ema_settings)

    # 將所有優化結果以附加模式寫入 CSV 檔案
    for result_batch in all_optimization_results:
        write_to_csv(output_csv_path, 'a', [result_batch])

    # 重新開啟 CSV 檔案以覆寫排序後的結果
    sorted_results = sorted(all_optimization_results, key=lambda x: (-x[4], -x[2]))
    write_to_csv(output_csv_path, 'w', sorted_results)


def filter_recent_trades(volume, months, output_csv_path, dataPath):
    optimal_ema_df = pd.read_csv(output_csv_path)
    recent_trades_stocks = []
    for filename in os.listdir(dataPath):
        stock_symbol = filename.split('.')[0]
        if int(stock_symbol) in optimal_ema_df['Stock'].values:
            optimal_ema = optimal_ema_df.loc[optimal_ema_df['Stock'] == int(stock_symbol), 'Optimal EMA'].iloc[0]
            data_path = os.path.join(dataPath, filename)
            data = pd.read_csv(data_path)
            data['Date'] = pd.to_datetime(data['Date'])
            # 正確解包trades和total_profit
            trades, _ = simulate_trades_with_volume(data, volume, ema_days=optimal_ema)
            recent_date = datetime.now() - timedelta(30 * months)
            # 確保trades是一個元組列表，其中第二個元素是可解析的日期
            recent_trades = [trade for trade in trades if pd.to_datetime(trade[1]) >= recent_date]
            if recent_trades:
                recent_trades_stocks.append(stock_symbol)
    return recent_trades_stocks


def filter_stocks_by_cap(stock_codes, data_path, cap_threshold=2e9):
    """
    篩選出股本超過特定閾值的股票代號。

    :param stock_codes: 股票代號列表。
    :param data_path: 包含股票數據檔案的目錄路徑。
    :param cap_threshold: 股本篩選閾值，預設為20億。
    :return: 超過股本閾值的股票代號列表。
    """
    stocks_with_large_cap = []

    for stock_code in stock_codes:
        file_name = f"{stock_code}.TW.csv"
        file_path = os.path.join(data_path, file_name)

        if os.path.exists(file_path):
            try:
                data = pd.read_csv(file_path)
                # 假設每個檔案的股本資訊是一致的，我們只查看第一行的股本
                if data['Shares Outstanding'].iloc[0] > cap_threshold:
                    stocks_with_large_cap.append(stock_code)
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    return stocks_with_large_cap


def BackTest2TradeSignals(volume, months, output_csv_path, dataPath, output_signalDir):
    """
    先看使用優化過的EMA數值產出的交易訊號，在三個月內有哪些股票有交易訊號，並進一步篩選流通股本>20億的股票，並輸出交易訊號CSV
    :param volume:
    :param months:
    :param output_csv_path:
    :param dataPath:
    :param output_signalDir:
    :return:
    """
    recent_stocks = filter_recent_trades(volume, months, output_csv_path, dataPath)
    print("使用最佳EMA數值近期交易訊號 : ", recent_stocks)
    # 篩選股本超過20億的股票代號
    stocks_over_20b = filter_stocks_by_cap(recent_stocks, dataPath)
    print("股本超過20億的股票代號:", stocks_over_20b)

    # 讀取 optimal_ema.csv 文件
    optimal_ema_df = pd.read_csv(output_csv_path)
    stocks_over_20b_int = [int(code) for code in stocks_over_20b]
    # 過濾 DataFrame
    filtered_optimal_ema_df = optimal_ema_df[optimal_ema_df['Stock'].isin(stocks_over_20b_int)]
    # 遍歷每個股票代碼及其對應的最佳 EMA 值
    for index, row in filtered_optimal_ema_df.iterrows():
        stock_code = int(row['Stock'])
        optimal_ema = int(row['Optimal EMA'])

        # 構建股票數據文件的路徑
        stock_file_path = os.path.join(dataPath, f'{stock_code}.TW.csv')
        if not os.path.exists(stock_file_path):
            print(f"Stock file for {stock_code} not found, skipping.")
            continue

        # 讀取股票數據
        stock_data = pd.read_csv(stock_file_path)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])

        # 獲取EMA數值的交易訊號
        trades, total_profit = simulate_trades_with_volume(stock_data, volume, optimal_ema)
        print("trades : ", trades)
        print("total_profit : ", total_profit)

        # 計算 EMA
        stock_data[f'EMA{optimal_ema}'] = stock_data['Close'].ewm(span=optimal_ema, adjust=False).mean()

        # 將DataFrame準備好以包含交易訊號和利潤
        stock_data[f'EMA{optimal_ema} Trade Signal'] = ''  # 在DataFrame中新增一列'Trade Signal'，用於存儲交易訊號
        stock_data[f'EMA{optimal_ema} Trade Profit'] = 0.0  # 在DataFrame中新增一列'Trade Profit'，用於存儲交易利潤

        # 透過交易資料迴圈更新DataFrame
        for trade in trades:
            trade_action, trade_date, trade_price, trade_profit, _ = trade  # 解包交易資料
            idx = stock_data[stock_data['Date'] == trade_date].index  # 找到交易日期對應的DataFrame索引
            if len(idx) > 0:  # 確保找到了匹配的索引
                stock_data.loc[idx, f'EMA{optimal_ema} Trade Signal'] = trade_action  # 更新DataFrame中交易訊號的值
                stock_data.loc[idx, f'EMA{optimal_ema} Trade Profit'] = trade_profit  # 更新DataFrame中交易利潤的值

        # 指定輸出CSV檔案路徑
        output_csv_path = os.path.join(output_signalDir, f'{stock_code}.TW.csv')  # 更新此路徑以指定要保存檔案的位置

        # 將更新後的DataFrame寫入新的CSV檔案
        stock_data.to_csv(output_csv_path, index=False)  # 將DataFrame寫入CSV檔案，不包含索引列

        print(f"已將交易和股票數據保存到 {output_csv_path}")  # 印出保存檔案的訊息


def plot_candlestick_chart(csv_file_path, ema_value=None):
    """
    繪製給定股票數據 CSV 檔案的蠟燭圖，並帶有基於指數移動平均線（EMA）的交易信號。

    參數:
    - csv_file_path: 股票數據 CSV 檔案的路徑。
    - ema_value: 可選參數；用於移動平均線的特定 EMA 值。如果為 None，腳本將嘗試查找 EMA 列。
    """
    # 從 CSV 檔案加載股票數據
    df = pd.read_csv(csv_file_path, parse_dates=['Date'], index_col='Date')

    # 創建兩個新的 DataFrame，一個用於買入信號，一個用於賣出信號，初始化為 NaN
    signals_buy = pd.DataFrame(index=df.index)
    signals_sell = pd.DataFrame(index=df.index)
    signals_buy['Signal'] = np.nan
    signals_sell['Signal'] = np.nan

    # 對於有買入 'Buy' 信號的行，將買入信號 DataFrame 對應的行更新為 df 中的收盤價（Close）
    # 對於有賣出 'Sell' 信號的行，將賣出信號 DataFrame 對應的行更新為 df 中的收盤價（Close）
    signals_buy.loc[df['EMA16 Trade Signal'] == 'Buy', 'Signal'] = df.loc[df['EMA16 Trade Signal'] == 'Buy', 'Close']
    signals_sell.loc[df['EMA16 Trade Signal'] == 'Sell', 'Signal'] = df.loc[df['EMA16 Trade Signal'] == 'Sell', 'Close']

    # 創建買入和賣出信號的圖層，分別使用不同的標記
    buy_markers = mpf.make_addplot(signals_buy['Signal'], type='scatter', markersize=100, marker='^', color='red',
                                   alpha=0.5)
    sell_markers = mpf.make_addplot(signals_sell['Signal'], type='scatter', markersize=100, marker='v', color='green',
                                    alpha=0.5)

    # 繪製圖表，包括買入和賣出信號
    mpf.plot(df, type='candle', addplot=[buy_markers, sell_markers], volume=True, style='charles')


def BackTest(dataPath):
    output_csv_path = r"D:\Temp\StockData\TW_STOCK_DATA\optimal_ema.csv"
    output_signalDir = r"D:\Temp\StockData\TW_STOCK_DATA\tradeSignals"

    if not os.path.exists(output_signalDir):
        os.makedirs(output_signalDir, exist_ok=False)

    volume = 1000  # 假設的交易量
    start_date = "2022-01-07"
    end_date = "2022-10-28"
    months = 3

    if not os.path.exists(output_csv_path):
        Backtest2Csv(volume, start_date, end_date, output_csv_path, dataPath)
    else:
        # BackTest2TradeSignals(volume, months, output_csv_path, dataPath, output_signalDir)
        plotStockList = os.listdir(output_signalDir)
        for plotStock in plotStockList:
            print(plotStock)
            plotStockPath = os.path.join(output_signalDir, plotStock)
            plot_candlestick_chart(plotStockPath)