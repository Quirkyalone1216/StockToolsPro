import os
import pandas as pd
import datetime
import requests
from PatternCrawl_CalAllPattern import CalAllPattern


def CrawlTWStockList():
    # 建立 URL 以獲取上市股票信息
    url = "https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=1&issuetype=1" \
          "&industry_code=&Page=1&chklike=Y"
    response = requests.get(url)

    # 使用 Pandas 讀取 HTML 表格，選擇所需字段，並設置列名
    listed = pd.read_html(response.text)[0]
    listed.columns = listed.iloc[0, :]
    listed = listed[["有價證券代號", "有價證券名稱", "市場別", "產業別", "公開發行/上市(櫃)/發行日"]]
    listed = listed.iloc[1:]

    # 創建股票代碼的完整格式，例如：2330.TW
    stock_nums = listed["有價證券代號"].apply(lambda x: str(x) + ".TW").tolist()

    """
    # 將股票代碼寫入文件
    with open('TW_Stock_Symbols.txt', 'w') as file:
        for symbol in stock_nums:
            file.write(symbol + '\n')
    """

    return stock_nums


def GetAllStockPattern():
    stock_nums = CrawlTWStockList()
    print(stock_nums)

    time_start = "18000101"
    # 獲取今天的日期
    today = datetime.date.today()

    # 使用timedelta增加一天
    time_end = (today + datetime.timedelta(days=1)).strftime("%Y%m%d")
    print(time_end)

    os.makedirs('result', exist_ok=True)

    Done = 0
    for stockID in stock_nums:
        if Done < 1:
            Done = CalAllPattern(time_start, time_end, stockID, 'Stock', '1d', 'result')
            if Done == 1:
                Done = 0
