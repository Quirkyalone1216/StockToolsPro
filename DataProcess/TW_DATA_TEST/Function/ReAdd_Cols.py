import pandas as pd
import os


def merge_data(merged_dir, predicted_dir, output_dir):
    """
    將合併數據中的“日期”和“信號”字段合併到預測數據中
    """
    os.makedirs(output_dir, exist_ok=True)

    for merged_file in os.listdir(merged_dir):
        merged_file_path = os.path.join(merged_dir, merged_file)
        predicted_file_path = os.path.join(predicted_dir, merged_file)

        # 確保預測數據中存在對應的文件
        if os.path.isfile(predicted_file_path):
            merged_df = pd.read_csv(merged_file_path)
            predicted_df = pd.read_csv(predicted_file_path)

            # Preparing the merged data
            merged_df = merged_df[['Date', 'Open', 'High', 'Low', 'Close', 'Signal']]

            merged_into_predicted = pd.merge(predicted_df, merged_df, on=['Open', 'High', 'Low', 'Close'],
                                             how='inner')

            output_file_path = os.path.join(output_dir, merged_file)
            merged_into_predicted.to_csv(output_file_path, index=False)

"""
# Define your directories here
merged_dir = 'path/to/your/TW_MergedData'
predicted_dir = 'path/to/your/TWPredictedData'
output_dir = 'path/to/your/output/directory'

merge_data(merged_dir, predicted_dir, output_dir)
"""
