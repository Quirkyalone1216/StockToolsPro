import pandas as pd
import os
import csv


def calculate_ema_profit(df, ema_span):
    """
    計算給定EMA跨度的累積盈利。
    """
    df['EMA'] = df['Close'].ewm(span=ema_span, adjust=False).mean()
    df['Pct_from_EMA'] = (df['Close'] - df['EMA']) / df['EMA'] * 100
    df['Buy'] = ((df['Pct_from_EMA'] > 3) | ((df['Pct_from_EMA'] > 1) & (df['Pct_from_EMA'].shift(1) > 1)))
    df['Sell'] = ((df['Pct_from_EMA'] < -3) | ((df['Pct_from_EMA'] < -1) & (df['Pct_from_EMA'].shift(1) < -1)))

    in_position = False
    buy_price = 0.0
    cumulative_profit = 0.0
    shares = 1000  # 假设每次交易1000股

    for _, row in df.iterrows():
        if row['Buy'] and not in_position:
            in_position = True
            buy_price = row['Close']
        elif row['Sell'] and in_position:
            sell_price = row['Close']
            profit_per_share = sell_price - buy_price
            cumulative_profit += profit_per_share * shares
            in_position = False

    return cumulative_profit


def find_best_ema_span(df):
    ema_spans = range(13, 68)
    cumulative_profits = {}

    for ema_span in ema_spans:
        cumulative_profit = calculate_ema_profit(df.copy(), ema_span)
        cumulative_profits[ema_span] = cumulative_profit

    best_ema_span = max(cumulative_profits, key=cumulative_profits.get)
    return best_ema_span, cumulative_profits[best_ema_span]


def generate_trade_signals(df, ema_span):
    df['EMA'] = df['Close'].ewm(span=ema_span, adjust=False).mean()
    df['Pct_from_EMA'] = (df['Close'] - df['EMA']) / df['EMA'] * 100
    df['Buy'] = ((df['Pct_from_EMA'] > 3) | ((df['Pct_from_EMA'] > 1) & (df['Pct_from_EMA'].shift(1) > 1)))
    df['Sell'] = ((df['Pct_from_EMA'] < -3) | ((df['Pct_from_EMA'] < -1) & (df['Pct_from_EMA'].shift(1) < -1)))
    trades = []
    in_position = False
    for index, row in df.iterrows():
        if row['Buy'] and not in_position:
            in_position = True
            trades.append({'Date': index, 'Action': 'Buy', 'Price': row['Close']})
        elif row['Sell'] and in_position:
            in_position = False
            sell_price = row['Close']
            buy_price = trades[-1]['Price']
            profit = sell_price - buy_price
            trades.append({'Date': index, 'Action': 'Sell', 'Price': sell_price, 'Profit': profit})

    return trades


def trade_signals(dataPath, log_file_path, csv_file_path):
    # 初始化日誌文件
    with open(log_file_path, 'w') as log_file:
        log_file.write('')

    # 初始化 CSV 文件並寫入表頭
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Stock', 'Best_EMA_Span', 'Interval_Profit', 'Cumulative_Profit'])

    # 遍歷股票文件並處理
    stockFileList = os.listdir(dataPath)
    results = []
    for stock_file in stockFileList:
        file_path = os.path.join(dataPath, stock_file)
        df_stock = pd.read_csv(file_path)
        df_stock['Date'] = pd.to_datetime(df_stock['Date'])
        df_stock.set_index('Date', inplace=True)

        best_ema_span, _ = find_best_ema_span(df_stock)
        tradeSignals = generate_trade_signals(df_stock, best_ema_span)

        last_trades = tradeSignals[-10:] if len(tradeSignals) > 10 else tradeSignals
        interval_profit = sum(trade.get('Profit', 0) for trade in last_trades if 'Profit' in trade)
        cumulative_profit = sum(trade.get('Profit', 0) for trade in tradeSignals if 'Profit' in trade)
        stock_name = stock_file.replace('.csv', '')

        stock_info = {
            'Stock': stock_name,
            'Best_EMA_Span': best_ema_span,
            'Interval_Profit': interval_profit,
            'Cumulative_Profit': cumulative_profit
        }

        results.append(stock_info)

        # 寫入 CSV 文件
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([stock_info['Stock'], stock_info['Best_EMA_Span'], stock_info['Interval_Profit'],
                                 stock_info['Cumulative_Profit']])

        # 寫入日誌文件和打印信息
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"股票名稱: {stock_info['Stock']}\n")
            log_file.write(f"最佳EMA跨度: {stock_info['Best_EMA_Span']}\n")
            log_file.write(f"間隔利潤: {stock_info['Interval_Profit']}\n")
            log_file.write(f"累計利潤: {stock_info['Cumulative_Profit']}\n")

            print(f"股票名稱: {stock_info['Stock']}")
            print(f"最佳EMA跨度: {stock_info['Best_EMA_Span']}")
            print(f"間隔利潤: {stock_info['Interval_Profit']}")
            print(f"累計利潤: {stock_info['Cumulative_Profit']}")

            total_profit = 0
            for signal in tradeSignals[-10:]:
                signal_date = signal['Date'].strftime('%Y-%m-%d')
                log_file.write(f"日期: {signal_date}, 行動: {signal['Action']}, 價格: {signal['Price']}")
                if 'Profit' in signal:
                    log_file.write(f", 利潤: {signal['Profit']}\n")
                    total_profit += signal['Profit']
                else:
                    log_file.write('\n')
            log_file.write(f"區間利潤: {total_profit}\n\n")

    return results


def EMA_Strategy(dataPath):
    # 處理所有股票檔案並收集信息
    log_file_path = r"D:\Temp\StockData\TW_STOCK_DATA\trade_signals_log.txt"
    csv_file_path = r"D:\Temp\StockData\TW_STOCK_DATA\stock_ema_results.csv"  # 新增的 CSV 檔案路徑
    results = trade_signals(dataPath, log_file_path, csv_file_path)  # 將 CSV 檔案路徑傳遞給 trade_signals()

    # 轉換為 DataFrame 並按區間利潤排序
    df_results = pd.DataFrame(results)
    df_sorted = df_results.sort_values(by='Interval_Profit', ascending=False)

    # 儲存到 CSV 檔案
    df_sorted.to_csv(csv_file_path, index=False)

    print(csv_file_path, df_sorted.head())  # 返回 CSV 檔案路徑和前幾行數據進行檢查
