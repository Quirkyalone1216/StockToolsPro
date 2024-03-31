import pandas as pd
import os


def process_and_save_stock_data(dir_path, save_path):
    # 获取解压后的文件列表
    files = os.listdir(dir_path)

    os.makedirs(save_path, exist_ok=True)

    # 处理每个CSV文件
    for file_name in files:
        file_path = os.path.join(dir_path, file_name)
        stock_id = file_name.split('.')[0]  # 从文件名获取股票代码

        # 读取并处理数据
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df['item_id'] = stock_id  # 添加item_id列
        df = df[['item_id', 'Date', 'Close']]  # 仅保留需要的列

        # 保存处理后的数据为新的CSV文件
        processed_file_path = os.path.join(save_path, file_name)
        if len(df) > 7:
            df.to_csv(processed_file_path, index=False)


# 路径设置（根据您的实际情况调整这些路径）
dir_path = r"D:\Temp\StockData\TW_STOCK_DATA\stock_data\Daily_K"
save_path = r"D:\Temp\StockData\TW_STOCK_DATA\stock_data\AutoGluon_TimeSeries"

# 执行数据处理
process_and_save_stock_data(dir_path, save_path)

print("数据处理完成，处理后的文件已保存。")
