import pandas as pd
import os


# 定義一個函數，將估算的交易量新增至原始CSV數據，並保存為新文件
def add_estimated_volumes_to_csv(data_path, newCsvPath):
    # 提供的時間和對應倍數的字典
    volume_ratios = {
        "09:15": 8,
        "09:30": 5,
        "09:45": 4,
        "10:00": 3,
        "10:15": 2.5,
        "10:30": 2.2,
        "10:45": 2,
        "11:00": 1.8,
        "11:15": 1.7,
        "11:30": 1.6,
        "11:45": 1.5,
        "12:00": 1.45,
        "12:15": 1.38,
        "12:30": 1.32,
        "12:45": 1.25,
        "13:00": 1.18,
        "13:15": 1.11,
        "13:30": 1
    }
    total_volume_ratio = sum(volume_ratios.values())

    files = os.listdir(data_path)
    for file in files:
        csv_path = os.path.join(data_path, file)

        # 讀取原始CSV數據
        data = pd.read_csv(csv_path)
        data['Time'] = pd.to_datetime(data['Date']).dt.strftime('%H:%M')
        data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')

        # 按日期分組，並加總交易量得到每日總交易量
        daily_volumes = data.groupby('Date')['Volume'].sum().reset_index(name='Total Daily Volume')

        # 為每個時間段計算估算交易量
        estimated_volumes = []
        for index, row in daily_volumes.iterrows():
            date = row['Date']
            total_daily_volume = row['Total Daily Volume']
            for time, ratio in volume_ratios.items():
                estimated_volume = (ratio / total_volume_ratio) * total_daily_volume
                estimated_volumes.append((date, time, estimated_volume))

        # 轉換為DataFrame
        estimated_volume_df = pd.DataFrame(estimated_volumes, columns=['Date', 'Time', 'Estimated Volume'])

        # 將估算的交易量合併至原始數據
        merged_data = pd.merge(data, estimated_volume_df, on=['Date', 'Time'], how='left')

        # 保存合併後的數據為新的CSV文件
        new_csv_path = os.path.join(newCsvPath, file)
        merged_data.to_csv(new_csv_path, index=False)


# 示例：運行函數的代碼，這裡需要替換成實際的CSV文件路徑
# csv_file_path = '/path/to/your/csv/file.csv'
# updated_csv_path = add_estimated_volumes_to_csv_v2(csv_file_path, volume_ratios, total_volume_ratio)
# print(f"Updated CSV file saved to: {updated_csv_path}")

def CalEstimatedVolumes(data_path, new_data_path):
    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)
    add_estimated_volumes_to_csv(data_path, new_data_path)
