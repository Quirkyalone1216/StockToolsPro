import pandas as pd
import os

from EMA_Strategy.TradeSignals import BackTest


def load_data(file_path):
    """從 CSV 檔案中讀取股票數據並轉換成 Pandas DataFrame 格式。"""
    data = pd.read_csv(file_path)
    return data


def apply_strategy(data, ema_period):
    """
    應用短期交易策略，計算指定天數的指數移動平均線（EMA）並根據條件生成買入和賣出信號。
    新增短線最高未實現獲利回吐30%停利賣出條件。
    """
    ema = data['Close'].ewm(span=ema_period, adjust=False).mean()
    data['EMA'] = ema

    # 初始化買入、賣出信號和最高價
    data['Buy_Signal'] = False
    data['Sell_Signal'] = False
    highest_price = 0  # 最高未實現獲利價格

    for i in range(1, len(data)):
        # 更新最高未實現獲利價格
        if data['Buy_Signal'][i - 1] and not data['Sell_Signal'][i - 1]:
            highest_price = max(highest_price, data['Close'][i])

        # 購買條件
        condition1 = data['Close'][i] > ema[i] * 1.03 and data['Close'][i - 1] < ema[i - 1]
        condition2 = i >= 2 and data['Close'][i] > ema[i] * 1.01 and data['Close'][i - 1] > ema[i - 1] * 1.01
        if condition1 or condition2:
            data.at[i, 'Buy_Signal'] = True
            highest_price = data['Close'][i]  # 重置最高價格為當前價格

        # 賣出條件：包括原始條件及新增的短線最高未實現獲利回吐30%
        condition3 = data['Close'][i] < ema[i] * 0.97 and data['Close'][i - 1] > ema[i - 1]
        condition4 = i >= 2 and data['Close'][i] < ema[i] * 0.99 and data['Close'][i - 1] < ema[i - 1] * 0.99
        condition5 = highest_price > 0 and data['Close'][i] < highest_price * 0.70  # 新增的停利賣出條件
        if condition3 or condition4 or condition5:
            data.at[i, 'Sell_Signal'] = True
            highest_price = 0  # 重置最高未實現獲利價格

    return data


def find_trades(data):
    """根據買入和賣出信號，找出交易點並計算每筆交易的利潤。"""
    trades = []
    current_buy_index = None
    for index, row in data.iterrows():
        if row['Buy_Signal'] and current_buy_index is None:
            current_buy_index = index
        elif row['Sell_Signal'] and current_buy_index is not None:
            trades.append((current_buy_index, index))
            current_buy_index = None
    # 計算每筆交易的利潤
    trade_results = []
    for buy_index, sell_index in trades:
        buy_price = data.loc[buy_index, 'Close']
        sell_price = data.loc[sell_index, 'Close']
        profit = (sell_price - buy_price) * 1000  # 假設每次交易 1000 股
        trade_results.append({'Buy_Date': data.loc[buy_index, 'Date'],
                              'Sell_Date': data.loc[sell_index, 'Date'],
                              'Profit': profit})
    return pd.DataFrame(trade_results)


def test_ema_settings(data, ema_range):
    """測試一系列不同的EMA設定，找出負利潤最少的最佳設定。"""
    best_setting = None
    min_negative_trades = float('inf')  # 初始化為無窮大
    optimal_profit = 0

    for ema_period in ema_range:
        # 應用短波段策略
        data_with_signals = apply_strategy(data.copy(), ema_period)
        # 找出交易並計算利潤
        trades_df = find_trades(data_with_signals)

        # 計算負利潤的交易數量
        try:
            negative_trades = trades_df[trades_df['Profit'] < 0].shape[0]
            # 更新最優設定，以負利潤最少為主要目標，若相同則以總利潤最高為次要考量
            if negative_trades < min_negative_trades or \
                    (negative_trades == min_negative_trades and trades_df['Profit'].sum() > optimal_profit):
                best_setting = ema_period
                min_negative_trades = negative_trades
                optimal_profit = trades_df['Profit'].sum()

            return best_setting, min_negative_trades, optimal_profit
        except KeyError as e:
            print(f"KeyError: {e}. Check if 'Profit' column exists in the DataFrame.")


def analyze_stock_data(file_path, start_date, end_date, ema_period):
    # 步驟1：載入股票數據
    data = load_data(file_path)

    # 篩選指定時間範圍的數據並重新設置索引
    filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)].reset_index(drop=True)

    # 步驟2：應用交易策略
    processed_data = apply_strategy(filtered_data, ema_period)

    # 步驟3：在指定時間範圍內尋找交易
    trade_results = find_trades(processed_data)

    try:
        # 計算統計數據
        positive_trades = trade_results[trade_results['Profit'] > 0].shape[0]
        negative_trades = trade_results[trade_results['Profit'] < 0].shape[0]
        total_interval_profit = trade_results['Profit'].sum()

        # 生成交易信號摘要
        buy_signals_count = processed_data['Buy_Signal'].sum()
        sell_signals_count = processed_data['Sell_Signal'].sum()

        # 編譯統計數據
        statistics = {
            'Positive_Trades': positive_trades,
            'Negative_Trades': negative_trades,
            'Total_Interval_Profit': total_interval_profit,
            'Buy_Signals': buy_signals_count,
            'Sell_Signals': sell_signals_count
        }

        return statistics
    except KeyError as e:
        print(f"KeyError: {e}. Check if 'Profit' column exists in the DataFrame.")


def trade_signals(dataPath, csv_file_path):
    # 取得股票資料目錄下的所有檔案列表
    stockDataList = os.listdir(dataPath)

    # 迭代處理每一個股票檔案
    for stock in stockDataList:
        file_path = os.path.join(dataPath, stock)
        data = load_data(file_path)  # 載入股票數據

        ema_range = range(5, 70)  # 測試EMA的範圍

        # 測試EMA設置
        optimal_ema, negative_trades, total_profit = test_ema_settings(data, ema_range)

        # 處理結果並寫入CSV文件
        if os.path.exists(csv_file_path):
            results_df = pd.read_csv(csv_file_path)
        else:
            results_df = pd.DataFrame(columns=['Stock', 'Optimal_EMA', 'Total_Profit'])

        new_row = pd.DataFrame([[stock, optimal_ema, total_profit]], columns=['Stock', 'Optimal_EMA', 'Total_Profit'])
        results_df = pd.concat([results_df, new_row], ignore_index=True)

        results_df.sort_values(by='Total_Profit', ascending=False, inplace=True)
        results_df.to_csv(csv_file_path, index=False)
        # 輸出結果
        print(f"{stock}: 最佳 EMA = {optimal_ema}, 負利潤交易數 = {negative_trades}, "
              f"總利潤 = {total_profit}")

        # 指定時間範圍回測
        start_date = '2022-01-01'
        end_date = '2023-01-01'
        analyze_stock_data(file_path, start_date, end_date, optimal_ema)


def EMA_Strategy(dataPath):
    csv_file_path = r"D:\Temp\StockData\TW_STOCK_DATA\optimal_ema_results.csv"
    """處理所有股票文件並收集相關信息。"""
    # trade_signals(dataPath, csv_file_path)  # 將 CSV 檔案路徑傳遞給 trade_signals()
    BackTest(dataPath)
