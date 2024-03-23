import os
import datetime
import requests
import concurrent.futures
from PatternCrawl_CalAllPattern import CalAllPattern
from bs4 import BeautifulSoup


# 爬取股票代號
def fetch_stock_symbols(letter):
    url = f"https://stock-screener.org/stock-list.aspx?alpha={letter}"
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/98.0.4758.102 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    symbols = [td.text.strip() for td in soup.find_all('td', {'width': '60', 'style': 'padding:5px;'}) if
               td.text.strip().isalpha()]
    return symbols


# 使用 fetch_stock_symbols 來獲取股票代號(MultiThreading)
def CrawUSStockList():
    all_symbols = []
    # 使用 ThreadPoolExecutor 並發爬取股票代號
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:

        future_to_letter = {executor.submit(fetch_stock_symbols, letter): letter for letter in
                            'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}

        for future in concurrent.futures.as_completed(future_to_letter):
            letter = future_to_letter[future]
            try:
                symbols = future.result()
                all_symbols.extend(symbols)
                print(f"已成功爬取以 {letter} 開頭的股票代號")
            except Exception as e:
                print(f"爬取以 {letter} 開頭的股票代號時發生錯誤: {e}")

    return all_symbols


def GetAllStockPattern():
    stock_nums = CrawUSStockList()
    print(stock_nums)

    time_start = "18000101"
    # 獲取今天的日期
    today = datetime.date.today()

    # 使用timedelta增加一天
    time_end = (today + datetime.timedelta(days=1)).strftime("%Y%m%d")
    print(time_end)

    os.makedirs('US_result', exist_ok=True)

    Done = 0
    for stockID in stock_nums:
        if Done < 1:
            Done = CalAllPattern(time_start, time_end, stockID, 'Stock', '1d', 'US_result')
            if Done == 1:
                Done = 0
