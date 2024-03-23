import pandas as pd
import os
from pathlib import Path

def process_file(file_path):
    try:
        df = pd.read_csv(file_path)

        # 轉換 'Date' 列為 datetime，去除時區，並僅保留日期
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(None).dt.date

        # 將修改後的 DataFrame 保存回同一文件
        df.to_csv(file_path, index=False)

        # print(f"Processed {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def Timezone2Date(directory_path):
    # 獲取目錄下的所有 CSV 文件
    csv_files = Path(directory_path).rglob('*.csv')

    # 處理每個文件
    for file_path in csv_files:
        process_file(file_path)


# 指定存有 CSV 文件的目錄路徑
# directory_path = r"/home/kenchen1216/StockTools/US_Stock/StockData"
