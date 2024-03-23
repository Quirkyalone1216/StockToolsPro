import os
import pandas as pd


def MergeOriginCSV(sort_result_dir, stock_data_dir, merged_data_dir, region):
    """
    # 假設的目錄路径，您需要根据实际情况进行调整
    sort_result_dir = '/home/kenchen1216/StockTools/US_Stock/TW_sort_result'
    stock_data_dir = '/home/kenchen1216/StockTools/US_Stock/TW_DATA'
    merged_data_dir = '/home/kenchen1216/StockTools/US_Stock/TWMergedData/'

    sort_result_dir = '/home/kenchen1216/StockTools/US_Stock/sort_result'
    stock_data_dir = '/home/kenchen1216/StockTools/US_Stock/StockData'
    merged_data_dir = '/home/kenchen1216/StockTools/US_Stock/MergedData/'
    """
    os.makedirs(merged_data_dir, exist_ok=True)

    # 取得兩個目錄中的股票代碼列表
    sort_result_stocks = [filename.split('.')[0] for filename in os.listdir(sort_result_dir) if '.' in filename]
    stock_data_stocks = [filename.split('.')[0] for filename in os.listdir(stock_data_dir) if '.' in filename]

    # 兩個目錄中的股票代碼交集
    common_stocks = set(sort_result_stocks).intersection(stock_data_stocks)

    stockCSV_name = ''
    # 股票代碼合併數據
    for stock_code in common_stocks:
        if region == 'TW':
            stockCSV_name = f"{stock_code}.TW.csv"
        elif region == 'US':
            stockCSV_name = f"{stock_code}.csv"

        sort_result_path = os.path.join(sort_result_dir, stockCSV_name)
        stock_data_path = os.path.join(stock_data_dir, stockCSV_name)
        merged_data_path = os.path.join(merged_data_dir, stockCSV_name)

        if not os.path.exists(sort_result_path) or not os.path.exists(stock_data_path):
            print(f"Skipping {stock_code}: File not found.")
            continue

        sort_result_df = pd.read_csv(sort_result_path)
        stock_data_df = pd.read_csv(stock_data_path)

        merged_df = pd.merge(stock_data_df, sort_result_df, on="Date", how="left")

        # 排除'Pattern', 'Signal'欄位空數據的那筆數據
        merged_df = merged_df.dropna(subset=['Pattern', 'Signal'], how='all')

        merged_df.to_csv(merged_data_path, index=False)

    print("Data merging completed. Merged files are saved in", merged_data_dir)
