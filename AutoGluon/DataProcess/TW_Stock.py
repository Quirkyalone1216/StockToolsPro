import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import shutil
import random

# 忽略pandas警告
warnings.filterwarnings('ignore')


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
    # 檢查日期列名稱
    possible_date_columns = ['Date', 'Datetime', 'date', 'datetime']
    for col_name in possible_date_columns:
        if col_name in df.columns:
            return col_name


def StockSignalPlot(data, tops, bottoms, stock):
    # 繪製股票價格圖
    data['Close'].plot()
    idx = data.index
    for top in tops:
        plt.plot(idx[top[1]], top[2], marker='o', color='green')

    for bottom in bottoms:
        plt.plot(idx[bottom[1]], bottom[2], marker='o', color='red')

    plt.title(f'{stock} closing price extreme')
    plt.legend(['Close'])
    plt.show()


def filterFilesSize(directory, size):
    OverSizeFiles = []
    files = os.listdir(directory)
    for file in files:
        filePath = os.path.join(directory, file)
        if os.path.getsize(filePath) > size:
            OverSizeFiles.append(file)
    return OverSizeFiles


def tradeSignalsGen(stockDataPath, outSignalsPath, stockList):
    for stock in stockList:
        """
        data = pd.read_csv('BTCUSDT86400.csv')
        data['Date'] = data['Date'].astype('datetime64[s]')
        """
        # print(f"Processing {stock}...")

        data = pd.read_csv(os.path.join(stockDataPath, stock))
        DateName = DateColName(data)

        data[DateName] = pd.to_datetime(data[DateName]).dt.tz_localize(None)
        data = clean_data(data)
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

        # StockSignalPlot(data, tops, bottoms, stock)

    print("Done")


def processData(df, DateName):
    # 根據 'Buy_Signal' 和 'Sell_Signal' 創建 'Signal' 欄位
    df['Signal'] = df.apply(
        lambda row: 'Buy' if pd.notna(row['Buy_Signal']) else ('Sell' if pd.notna(row['Sell_Signal']) else 'Hold'),
        axis=1)

    # # 將 'Signal' 映射為數值：買入 = 1，賣出 = -1，持有 = 0
    # signal_mapping = {'買入': 1, '賣出': -1, '持有': 0}
    # df['Signal'] = df['Signal'].map(signal_mapping)

    # 刪除原始的 'Buy_Signal' 和 'Sell_Signal' 欄位
    df = df.drop(['Buy_Signal', 'Sell_Signal'], axis=1)

    # 如果不需要，可選擇移除 'Unnamed: 0' 和 'Date' 欄位
    df = df.drop(['Unnamed: 0', DateName], axis=1)

    return df


def TrainDataSplit(source_folder, target_root, size_limit_mb=600):
    """
    將處理後的數據分割為大小不超過指定大小的多個目錄
    """
    # 計算大小限制的字節數
    size_limit_bytes = size_limit_mb * 1024 * 1024

    # 獲取所有檔案及其大小
    files = []
    for dirpath, dirnames, filenames in os.walk(source_folder):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            filesize = os.path.getsize(filepath)
            files.append((filepath, filesize))

    # 隨機打亂檔案列表
    random.shuffle(files)

    subfolder_index = 1
    while files:
        current_folder = os.path.join(target_root, f'subfolder_{subfolder_index}')
        os.makedirs(current_folder, exist_ok=True)
        current_size = 0

        # 檔案移動迴圈
        for (filepath, filesize) in list(files):  # 使用list進行循環以允許從原列表中移除項目
            if current_size + filesize <= size_limit_bytes:
                shutil.move(filepath, current_folder)
                current_size += filesize
                files.remove((filepath, filesize))

        subfolder_index += 1

        # 如果當前資料夾未達到大小限制，且沒有更多檔案可添加，則結束迴圈
        if current_size < size_limit_bytes and sum(fsize for _, fsize in files) < size_limit_bytes:
            break


def trainDataPreproc():
    # try:
    # 設定路徑
    stockDataPath = r"D:\Temp\StockData\TW_STOCK_DATA\tradeSignals"
    trainDataPath = r"D:\Temp\StockData\TW_STOCK_DATA\VaildData"
    targetDataPath = r"D:\Temp\StockData\TW_STOCK_DATA\TargetData"
    bytes_of_size = 600

    if not os.path.exists(trainDataPath):
        os.makedirs(trainDataPath, exist_ok=False)

    signalsFiles = os.listdir(stockDataPath)
    for file in signalsFiles:
        data = pd.read_csv(os.path.join(stockDataPath, file))
        # 獲取日期列名稱
        DateName = DateColName(data)
        # 把 'Buy_Signal' 和 'Sell_Signal' 列合併成 'Signal' 列
        data = processData(data, DateName)

        # 輸出新的CSV文件
        output_path = os.path.join(trainDataPath, file)
        data.to_csv(output_path, index=False)

    # TrainDataSplit(trainDataPath, targetDataPath, bytes_of_size)
    # except Exception as e:
    #     print(e)


def TW_StockSignal():
    try:
        # 設定路徑
        stockDataPath = r"D:\Temp\StockData\TW_STOCK_DATA\stock_data\Daily_K"
        outSignalsPath = r"D:\Temp\StockData\TW_STOCK_DATA\tradeSignals"

        if not os.path.exists(outSignalsPath):
            os.makedirs(outSignalsPath, exist_ok=False)

        # size = 40 * 1024    # 40KB
        # stockList = filterFilesSize(stockDataPath, size)
        stockList = os.listdir(stockDataPath)
        tradeSignalsGen(stockDataPath, outSignalsPath, stockList)
        trainDataPreproc()

    except Exception as e:
        print(e)


def clean_data(df):
    # 刪除 'Open' 為 0 的列
    df = df[['Open'] != 0]

    # 辨識 'Volume' 為 0 的列
    df['Zero_Volume'] = (df['Volume'] == 0)
    
    # 尋找連續三天或更多天 'Volume' 為 0 的序列
    df['Group'] = (df['Zero_Volume'] != df['Zero_Volume'].shift()).cumsum()
    df['Count_In_Group'] = df.groupby('Group')['Zero_Volume'].transform('sum')
    
    # 篩選出 'Count_In_Group' 為 3或更多的群組
    df = df[(~df['Zero_Volume']) | (df['Count_In_Group'] < 3)]
    
    # 刪除暫時性欄位和 'Dividends'、'Stock Splits'、'Symbol' 欄位
    df.drop(columns=['Zero_Volume', 'Group', 'Count_In_Group', 'Dividends', 'Stock Splits', 'Symbol'], inplace=True)
    
    return df


if __name__ == "__main__":
    TW_StockSignal()
