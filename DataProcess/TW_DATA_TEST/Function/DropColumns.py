import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime


def process_csv_file_cpu(file_path, region):
    data = pd.read_csv(file_path)

    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by='Date', ascending=True)

    # 日期特徵
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek

    # 計算從數據集開始的天數
    earliest_date = data['Date'].min()
    data['DaysFromStart'] = (data['Date'] - earliest_date).dt.days

    # 要移除的Column(台股、美股)
    common_columns_to_drop = ['Date', 'Dividends', 'Stock Splits']
    us_specific_columns = ['Symbol', 'Capital Gains']
    tw_specific_columns = ['Adj Close', 'Shares Outstanding']

    if region == 'US':
        data['Symbol'] = data['Symbol'].astype(str)
        columns_to_drop = common_columns_to_drop + us_specific_columns
    elif region == 'TW':
        columns_to_drop = common_columns_to_drop + tw_specific_columns

    # 確保每個columns都被移除
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]

    data = data.drop(columns=columns_to_drop).dropna()

    # 排除開盤價數值為0的數據
    data = data[data['Open'] != 0]

    # Pattern encoding
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(data['Pattern'])  # Fallback to pandas
    data['Pattern'] = encoded

    return data


def DropColumns(merged_data_dir, processed_data_dir, region):
    """
    # 遍歷 MergedData 目錄下的所有 CSV 文件
    merged_data_dir = '/home/kenchen1216/StockTools/US_Stock/MergedData'
    processed_data_dir = '/home/kenchen1216/StockTools/US_Stock/AutoSklearnData'
    """

    csv_files = [f for f in os.listdir(merged_data_dir) if f.endswith('.csv')]

    os.makedirs(processed_data_dir, exist_ok=True)

    for file_name in csv_files:
        file_path = os.path.join(merged_data_dir, file_name)
        processed_data = process_csv_file_cpu(file_path, region)  # 處理數據
        # 保存處理後的文件
        save_path = os.path.join(processed_data_dir, file_name)
        processed_data.to_csv(save_path, index=False)
