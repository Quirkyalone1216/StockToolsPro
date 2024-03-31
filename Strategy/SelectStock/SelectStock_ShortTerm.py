import pandas as pd
import os


def ShortTerm(daily_path, estimated_volumes_path):
    # 載入預估交易量數據
    estimated_volumes = pd.read_csv(estimated_volumes_path)
    met_stock = []

    # 遍歷所有股票檔案
    for file in os.listdir(daily_path):
        file_path = os.path.join(daily_path, file)
        stock_data = pd.read_csv(file_path)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])

        # 確保數據是按日期升序排序的，這樣最後一行就是最近的數據
        stock_data.sort_values('Date', ascending=True, inplace=True)

        # 從 estimated_volumes 獲取最新預估量
        matching_volumes = estimated_volumes[estimated_volumes['File'] == file]['Latest Estimated Volume for 09:15']
        if not matching_volumes.empty:
            latest_volume = matching_volumes.iloc[0]

        # 確定最後一行數據的索引
        last_row_index = stock_data.index[-1]

        # 計算所需的交易量
        prev_day_volume = stock_data.iloc[last_row_index - 1]['Volume']
        prev_5_days_avg_volume = stock_data.iloc[max(0, last_row_index - 5):last_row_index]['Volume'].mean()
        prev_quarter_avg_volume = stock_data.iloc[max(0, last_row_index - 65):last_row_index]['Volume'].mean()

        # 檢查是否符合條件
        condition_met = latest_volume > 2 * prev_day_volume or \
                        latest_volume > 2 * prev_5_days_avg_volume or \
                        latest_volume > 2 * prev_quarter_avg_volume

        # 如果符合條件，則加入到 qualifying_stocks
        if condition_met:
            met_stock.append(file)

    # 顯示符合條件的股票
    print(met_stock)
    return met_stock
