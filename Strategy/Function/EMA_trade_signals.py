import pandas as pd
import os


def calculate_ema_profit(df, ema_span):
    """
    计算给定EMA跨度的累积盈利。
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
    in_position = False
    buy_price = 0.0
    trades = []
    for date, row in df.iterrows():
        if row['Buy'] and not in_position:
            in_position = True
            buy_price = row['Close']
            trades.append({'Date': date, 'Action': 'Buy', 'Price': buy_price})
        elif row['Sell'] and in_position:
            sell_price = row['Close']
            profit = sell_price - buy_price
            trades.append({'Date': date, 'Action': 'Sell', 'Price': sell_price, 'Profit': profit})
            in_position = False

    # Handle the case where the last action was a buy without a corresponding sell
    if in_position:
        # Assuming the last close price as the sell price
        last_close_price = df.iloc[-1]['Close']
        profit = last_close_price - buy_price
        last_date = df.index[-1]
        trades.append({'Date': last_date, 'Action': 'Sell', 'Price': last_close_price, 'Profit': profit})

    return trades


def trade_signals(dataPath):
    stockFileList = os.listdir(dataPath)
    for file in stockFileList:
        print(file.replace('.csv', ''))
        sample_file_path = os.path.join(dataPath, file)
        df_sample = pd.read_csv(sample_file_path)
        df_sample['Date'] = pd.to_datetime(df_sample['Date'])
        df_sample.set_index('Date', inplace=True)

        # 找出最佳EMA跨度
        best_ema_span, best_ema_profit = find_best_ema_span(df_sample)
        print(f"最佳 EMA 跨度: {best_ema_span}, 累計利潤: {best_ema_profit}")

        # 生成交易信號
        tradeSignals = generate_trade_signals(df_sample, best_ema_span)
        total_profit = 0  # 初始化總利潤
        for signal in tradeSignals[-5:]:  # 顯示最後五個交易信號作為示例
            signal_date = signal['Date'].strftime('%Y-%m-%d')  # 將日期格式化為年-月-日格式
            print(f"日期: {signal_date}, 行動: {signal['Action']}, 價格: {signal['Price']}", end="")
            if 'Profit' in signal:
                print(f", 利潤: {signal['Profit']}")
                total_profit += signal['Profit']  # 將利潤累加到總利潤中
            else:
                print()
        print(f"區間利潤: {total_profit}\n")  # 輸出總利潤
