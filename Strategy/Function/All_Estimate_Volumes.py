import os
import pandas as pd


def get_csv_file_paths(directory_path):
    """獲取給定目錄下所有CSV文件的路徑"""
    return [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.csv')]


def get_latest_estimated_volume_for_0915(csv_file_path):
    """從給定的CSV文件中提取最新一天09:15的預估交易量值"""
    try:
        df = pd.read_csv(csv_file_path)
        filtered_df = df[df['Time'] == '09:15']
        if filtered_df.empty:
            return None
        latest_date = filtered_df['Date'].max()
        latest_row = filtered_df[filtered_df['Date'] == latest_date]
        return latest_row['Estimated Volume'].values[0]
    except Exception as e:
        print(f"處理文件 {csv_file_path} 時發生錯誤: {e}")
        return None


def save_results_to_csv(results, output_path):
    """將結果保存到CSV文件"""
    pd.DataFrame(list(results.items()), columns=['File', 'Latest Estimated Volume for 09:15']).to_csv(output_path,
                                                                                                      index=False)


def AllEstimateVolumes(data_path):
    csv_file_paths = get_csv_file_paths(data_path)

    # 提取數據
    all_results = {}
    for csv_file_path in csv_file_paths:
        estimated_volume = get_latest_estimated_volume_for_0915(csv_file_path)
        if estimated_volume is not None:
            all_results[os.path.basename(csv_file_path)] = estimated_volume

    # 保存結果
    results_csv_path = r'D:\Temp\StockData\TW_STOCK_DATA\estimated_volumes_latest.csv'
    save_results_to_csv(all_results, results_csv_path)
    print(f"結果保存到 {results_csv_path}")
