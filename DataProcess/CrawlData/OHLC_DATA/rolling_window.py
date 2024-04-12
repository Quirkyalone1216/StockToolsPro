import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import scipy


# 檢查是否在當前索引處檢測到局部高點
def rw_top(data: np.array, curr_index: int, order: int) -> bool:
    if curr_index < order * 2 + 1:
        return False

    top = True
    k = curr_index - order
    v = data[k]
    for i in range(1, order + 1):
        if data[k + i] > v or data[k - i] > v:
            top = False
            break

    return top


# 檢查是否在當前索引處檢測到局部低點
def rw_bottom(data: np.array, curr_index: int, order: int) -> bool:
    if curr_index < order * 2 + 1:
        return False

    bottom = True
    k = curr_index - order
    v = data[k]
    for i in range(1, order + 1):
        if data[k + i] < v or data[k - i] < v:
            bottom = False
            break

    return bottom


def rw_extremes(data: np.array, order: int):
    # 滾動窗口局部高點和低點
    tops = []
    bottoms = []
    for i in range(len(data)):
        if rw_top(data, i, order):
            # top[0] = 確認索引
            # top[1] = 高點索引
            # top[2] = 高點價格
            top = [i, i - order, data[i - order]]
            tops.append(top)

        if rw_bottom(data, i, order):
            # bottom[0] = 確認索引
            # bottom[1] = 低點索引
            # bottom[2] = 低點價格
            bottom = [i, i - order, data[i - order]]
            bottoms.append(bottom)

    return tops, bottoms


def DateColName(df):
    possible_date_columns = ['Date', 'Datetime', 'date', 'datetime']
    for col_name in possible_date_columns:
        if col_name in df.columns:
            return col_name


def TW_StockSignal():
    stockDataPath = r"D:\Temp\StockData\TW_STOCK_DATA\stock_data\Daily_K"

    outSignalsPath = r"D:\Temp\StockData\TW_STOCK_DATA\tradeSignals"
    if not os.path.exists(outSignalsPath):
        os.makedirs(outSignalsPath, exist_ok=False)

    stockList = os.listdir(stockDataPath)
    for stock in stockList:
        """
        data = pd.read_csv('BTCUSDT86400.csv')
        data['Date'] = data['Date'].astype('datetime64[s]')
        """
        data = pd.read_csv(os.path.join(stockDataPath, stock))
        DateName = DateColName(data)

        data[DateName] = pd.to_datetime(data[DateName]).dt.tz_localize(None)
        _, bottoms = rw_extremes(data['Close'].to_numpy(), 20)
        tops, _ = rw_extremes(data['Close'].to_numpy(), 20)

        # 標註買賣訊號
        data['Buy_Signal'] = None
        data['Sell_Signal'] = None
        for top in tops:
            data.at[data.index[top[1]], 'Sell_Signal'] = 'Sell'
        for bottom in bottoms:
            data.at[data.index[bottom[1]], 'Buy_Signal'] = 'Buy'

        # 輸出新的CSV文件
        output_path = os.path.join(outSignalsPath, stock)
        data.to_csv(output_path)

        """
        data['Close'].plot()
        idx = data.index
        for top in tops:
            plt.plot(idx[top[1]], top[2], marker='o', color='green')

        for bottom in bottoms:
            plt.plot(idx[bottom[1]], bottom[2], marker='o', color='red')

        plt.title(f'{stock} closing price extreme')
        plt.legend(['Close'])
        plt.show()
        """


# 定義處理每個檔案並找到有交易訊號的股票的函式
def check_signals(file_path, day):
    df = pd.read_csv(file_path)
    # 確保日期以降序排序以獲得最後指定天數
    DateName = DateColName(df)

    df[DateName] = pd.to_datetime(df[DateName]).dt.tz_localize(None)
    df.sort_values(DateName, ascending=False, inplace=True)

    # 提取最後指定天數
    last_days = df.head(day)

    # 檢查買入和賣出信號
    buy_signal = last_days['Buy_Signal'].notna().any()
    sell_signal = last_days['Sell_Signal'].notna().any()

    return buy_signal, sell_signal


def extract_signal_details(file_path, day):
    df = pd.read_csv(file_path)
    DateName = DateColName(df)
    df[DateName] = pd.to_datetime(df[DateName]).dt.tz_localize(None)
    df.sort_values(DateName, ascending=False, inplace=True)
    last_days = df.head(day)
    buy_signals = last_days[last_days['Buy_Signal'].notna()][[DateName, 'Close', 'Buy_Signal']]
    sell_signals = last_days[last_days['Sell_Signal'].notna()][[DateName, 'Close', 'Sell_Signal']]
    return buy_signals, sell_signals


def format_output(buy_list, sell_list, buy_details, sell_details, look_back):
    # 格式化股票清單
    buy_list_str = ", ".join(buy_list)
    sell_list_str = ", ".join(sell_list)

    buy_output = f"近{look_back}天買入股票清單: [{buy_list_str}]\n"
    sell_output = f"近{look_back}天賣出股票清單: [{sell_list_str}]\n"

    # 格式化買入信號詳細資訊
    buy_signals_output = "買入信號詳細資訊:\n"
    for stock, signals in buy_details.items():
        buy_signals_output += f"- 股票代碼 {stock}:\n"
        for _, row in signals.iterrows():
            # date = row['Date'].date()  # 直接使用Timestamp對象的date方法
            date = row[DateColName(signals)]  # 'Series' object
            close_price = f"{row['Close']:.2f}"  # 格式化收盤價至小數點後兩位
            buy_signals_output += f"  日期: {date}, 收盤價: {close_price}, 訊號: {row['Buy_Signal']}\n"

    # 格式化賣出信號詳細資訊
    sell_signals_output = "賣出信號詳細資訊:\n"
    for stock, signals in sell_details.items():
        sell_signals_output += f"- 股票代碼 {stock}:\n"
        for _, row in signals.iterrows():
            # date = row['Date'].date()  # 直接使用Timestamp對象的date方法
            date = row[DateColName(signals)]  # 'Series' object
            close_price = f"{row['Close']:.2f}"  # 格式化收盤價至小數點後兩位
            sell_signals_output += f"  日期: {date}, 收盤價: {close_price}, 訊號: {row['Sell_Signal']}\n"

    # 結合所有輸出並返回
    return buy_output + sell_output + buy_signals_output + sell_signals_output


def recentSignals(days):
    # 檢查最後指定天數的買入和賣出信號
    signalsPath = r"D:\Temp\StockData\TW_STOCK_DATA\tradeSignals"
    stockList = os.listdir(signalsPath)
    look_back = days
    buy_stock = []
    sell_stock = []

    for stock in stockList:
        file_path = os.path.join(signalsPath, stock)
        buy_signal, sell_signal = check_signals(file_path, look_back)

        if buy_signal:
            buy_stock.append(stock.split(".")[0])
        if sell_signal:
            sell_stock.append(stock.split(".")[0])

    # print(f"近{look_back}天買入股票清單 : ", buy_stock)
    # print(f"近{look_back}天賣出股票清單 : ", sell_stock)

    # 提取買入和賣出信號的詳細資訊
    buy_signals_details = {}
    sell_signals_details = {}
    for stock in buy_stock:
        file_path = os.path.join(signalsPath, stock + ".TW.csv")
        buy_signals, _ = extract_signal_details(file_path, look_back)
        if not buy_signals.empty:
            buy_signals_details[stock] = buy_signals

    for stock in sell_stock:
        file_path = os.path.join(signalsPath, stock + ".TW.csv")
        _, sell_signals = extract_signal_details(file_path, look_back)
        if not sell_signals.empty:
            sell_signals_details[stock] = sell_signals

    # return buy_signals_details, sell_signals_details
    # print("買入信號詳細資訊 : ", buy_signals_details)
    # print("賣出信號詳細資訊 : ", sell_signals_details)

    # 使用format_output函式格式化輸出
    output = format_output(buy_stock, sell_stock, buy_signals_details, sell_signals_details, look_back)
    return output


if __name__ == "__main__":
    """
    data = pd.read_csv('BTCUSDT86400.csv')
    data['date'] = data['date'].astype('datetime64[s]')

    tops, bottoms = rw_extremes(data['close'].to_numpy(), 10)

    data['close'].plot()
    idx = data.index
    for top in tops:
        plt.plot(idx[top[1]], top[2], marker='o', color='green')

    for bottom in bottoms:
        plt.plot(idx[bottom[1]], bottom[2], marker='o', color='red')

    plt.show()
    """
    TW_StockSignal()

# Scipy 實現（更快，但請小心不要使用未來數據）
# import scipy
# arr = data['close'].to_numpy()
# bottoms = scipy.signal.argrelextrema(arr, np.less, order=3)
# tops = scipy.signal.argrelextrema(arr, np.greater, order=3)
