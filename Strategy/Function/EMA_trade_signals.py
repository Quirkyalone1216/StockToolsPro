import pandas as pd
import os


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


def trade_signals(dataPath, log_file_path):
    stockFileList = os.listdir(dataPath)
    results = []
    with open(log_file_path, 'w') as log_file:
        log_file.write('')

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

        # 開啟日誌文件以附加模式寫入日誌
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"股票名稱: {stock_info['Stock']}\n")
            log_file.write(f"最佳EMA跨度: {stock_info['Best_EMA_Span']}\n")
            log_file.write(f"間隔利潤: {stock_info['Interval_Profit']}\n")
            log_file.write(f"累計利潤: {stock_info['Cumulative_Profit']}\n")

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
    results = trade_signals(dataPath, log_file_path)

    # 轉換為DataFrame並按區間利潤排序
    df_results = pd.DataFrame(results)
    df_sorted = df_results.sort_values(by='Interval_Profit', ascending=False)

    # 儲存到CSV檔案
    csv_file_path = r"D:\Temp\StockData\TW_STOCK_DATA\stock_ema_results.csv"
    df_sorted.to_csv(csv_file_path, index=False)

    print(csv_file_path, df_sorted.head())  # 返回CSV檔案路徑和前幾行數據進行檢查
