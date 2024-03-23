import requests
import pandas as pd
from io import StringIO
import datetime
import os
import shutil


def SaveDataPath():
    # Path = input("請輸入臺灣證交所股票歷史股價儲存路徑: ").strip('"')
    Path = r"TW_OHLC_DATA"
    os.makedirs(Path, exist_ok=True)
    return Path


def getCurrentPath(dataPath):
    # 創建stock_data資料夾的路徑
    # os.path.abspath，需要用絕對路徑，因打單個文件包成.exe路徑會跑到Temp資料夾內
    stock_data_path = os.path.abspath(os.path.join(dataPath, 'stock_data'))

    # 如果資料夾不存在，則創建它
    if not os.path.exists(stock_data_path):
        os.makedirs(stock_data_path)

    return stock_data_path


def getCsvPath(stock_id, dataPath):
    # 獲取當前檔案的絕對路徑
    stock_data_path = os.path.abspath(os.path.join(dataPath, 'stock_data'))

    # 如果資料夾不存在，則創建它
    if not os.path.exists(stock_data_path):
        os.makedirs(stock_data_path)

    # 根據stock_id創建csv檔案的路徑
    address = os.path.join(stock_data_path, f"{stock_id}.csv")

    return address


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


# 讀取或寫入股票歷史資料至CSV檔案
def stock_data(stock_id, time_start, time_end, dataPath):
    days = 24 * 60 * 60  # 一天有86400秒
    initial = datetime.datetime.strptime('1900-01-01', '%Y-%m-%d')
    start = datetime.datetime.strptime(time_start, '%Y-%m-%d')
    end = datetime.datetime.strptime(time_end, '%Y-%m-%d')
    period1 = start - initial
    period2 = end - initial
    s1 = period1.days * days
    s2 = period2.days * days
    url = "https://query1.finance.yahoo.com/v7/finance/download/" + stock_id + "?period1=" + str(
        s1) + "&period2=" + str(s2) + "&interval=1d&events=history&includeAdjustedClose=true"
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/98.0.4758.102 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    df = pd.read_csv(StringIO(response.text), index_col="Date", parse_dates=["Date"])
    address = getCsvPath(stock_id, dataPath)
    if os.path.isfile(address):
        df_new = pd.read_csv(address, index_col="Date", parse_dates=["Date"])
        if time_start not in df_new.index:
            df_new = df_new.append(df)
            df_new.to_csv(address, encoding='utf-8')
            print("已更新到最新資料")
        else:
            print("已是最新資料，無需更新")
    else:
        df.to_csv(address, encoding='utf-8')
        print("此為新資料，已創建csv檔")


def GetTWStock_OHLC_Data():
    # 臺灣證交所股票歷史股價儲存路徑
    dataPath = SaveDataPath()

    # 刪除stock_data資料夾中的所有檔案
    delStockData(dataPath)

    # 建立URL以獲取上市股票資訊
    url = "https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=1&issuetype=1" \
          "&industry_code=&Page=1&chklike=Y"
    response = requests.get(url)

    # 使用Pandas讀取HTML表格，選取所需欄位，並設定列名
    listed = pd.read_html(response.text)[0]
    listed.columns = listed.iloc[0, :]
    listed = listed[["有價證券代號", "有價證券名稱", "市場別", "產業別", "公開發行/上市(櫃)/發行日"]]
    listed = listed.iloc[1:]
    print(listed)
    with open('TW_STOCK_LIST.txt', 'w', encoding='utf-8') as file:
        file.write(listed.to_string())

    # 創建股票代號的完整格式，例如：2330.TW
    stock_1 = listed["有價證券代號"]
    stock_num = stock_1.apply(lambda x: str(x) + ".TW")
    print(stock_num)

    # 設定起始日期與結束日期
    time_start = "1800-01-01"
    time_end = datetime.date.today().strftime("%Y-%m-%d")

    for i in stock_num:
        try:
            stock_data(i, time_start, time_end, dataPath)
            print(i + "\n" + "successful")
        except:
            print(i + "\n" + "fail")


if __name__ == '__main__':
    GetTWStock_OHLC_Data()
