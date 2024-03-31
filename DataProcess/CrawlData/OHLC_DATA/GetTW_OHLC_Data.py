import yfinance as yf
import pandas as pd
import os
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter
import requests
import datetime
import shutil


# 定義一個具有緩存和限速功能的Session類
class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass


"""
# 創建一個session實例，配置了每5秒最多2次請求的限速，以及使用SQLite作為緩存後端
session = CachedLimiterSession(
    limiter=Limiter(RequestRate(2, Duration.SECOND * 30)),
    bucket_class=MemoryQueueBucket,
    backend=SQLiteCache("yfinance.cache"),
)
"""


def delStockData(dataPath):
    folder_path = getCurrentPath(dataPath)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def getCurrentPath(dataPath):
    # 創建stock_data資料夾的路徑
    # os.path.abspath，需要用絕對路徑，因打單個文件包成.exe路徑會跑到C槽Temp資料夾內

    # stock_data_path = os.path.abspath(os.path.join(dataPath, 'stock_data'))
    stock_data_path = os.path.abspath(r"D:\Temp\StockData\TW_STOCK_DATA\stock_data")

    # 如果資料夾不存在，則創建它
    if not os.path.exists(stock_data_path):
        os.makedirs(stock_data_path)

    return stock_data_path


def SaveDataPath(K_types):
    TotalDir = []
    Path = os.path.abspath(r"D:\Temp\StockData\TW_STOCK_DATA\stock_data")
    for K_type in K_types:
        K_type_Dir = os.path.abspath(os.path.join(Path, K_type))
        TotalDir.append(K_type_Dir)
        os.makedirs(Path, exist_ok=True)

    return TotalDir


def getCsvPath(stock_id, dataPath):
    stock_data_path = os.path.abspath(dataPath)
    if not os.path.exists(stock_data_path):
        os.makedirs(stock_data_path)
    address = os.path.join(stock_data_path, f"{stock_id}.csv")
    return address


def getTWStockList():
    url = "https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=1&issuetype=1" \
          "&industry_code=&Page=1&chklike=Y"
    response = requests.get(url)
    listed = pd.read_html(response.text)[0]
    listed.columns = listed.iloc[0, :]
    listed = listed[["有價證券代號", "有價證券名稱", "市場別", "產業別", "公開發行/上市(櫃)/發行日"]]
    listed = listed.iloc[1:]
    listed["有價證券代號"] = listed["有價證券代號"].apply(lambda x: str(x) + ".TW")
    return listed


def downloadTWStockData(stock_id, time_start, time_end, dataPath, K_type):
    # 使用配置好的session來創建yfinance Ticker對象
    # stock = yf.Ticker(stock_id, session=session)
    stock = yf.Ticker(stock_id)
    # print(stock.info)

    df = stock.history(start=time_start, end=time_end, interval=K_type)
    # 取得股本數據
    shares_outstanding = stock.info['sharesOutstanding']
    df['Shares Outstanding'] = shares_outstanding
    address = getCsvPath(stock_id, dataPath)
    if not os.path.isfile(address):
        df.to_csv(address)
        # print(f"{stock_id}: 新資料，已創建csv檔")
    else:
        print(f"{stock_id}: csv檔案已存在")


def GetTWStock_OHLC_Data(time_start, time_end, time_start_15m):
    K_type = ['Daily_K', 'Weekly_K', '15Minutes_K']
    K_Data_type = ['1d', '1wk', '15m']
    K_total_dir = SaveDataPath(K_type)
    for K_dir in K_total_dir:
        delStockData(K_dir)

    listed = getTWStockList()
    """
    time_start = "1900-01-01"
   
    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=2)
    time_end = tomorrow.strftime("%Y-%m-%d")
    
    time_end = "2023-07-08"
    """

    for index, row in listed.iterrows():
        stock_id = row["有價證券代號"]
        try:
            downloadTWStockData(stock_id, time_start, time_end, K_total_dir[0], K_Data_type[0])
            downloadTWStockData(stock_id, time_start, time_end, K_total_dir[1], K_Data_type[1])
            downloadTWStockData(stock_id, time_start_15m, time_end, K_total_dir[2], K_Data_type[2])

        except Exception as e:
            print(f"{stock_id}: 下載失敗，原因: {e}")


if __name__ == '__main__':
    time_start = "1900-01-01"
    today = datetime.date.today()

    time_start_15m = today - datetime.timedelta(days=10)
    tomorrow = today + datetime.timedelta(days=2)
    time_end = tomorrow.strftime("%Y-%m-%d")

    # time_end = "2023-08-03"
    GetTWStock_OHLC_Data(time_start, time_end, time_start_15m)
