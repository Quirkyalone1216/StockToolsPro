from Function.MergeOriginCSV import MergeOriginCSV
from Function.DropColumns import DropColumns
from Function.FeaturesProcess import FeaturesProcess
from Function.ReAdd_Cols import merge_data
from Function.CalModelAccuracy import calculate_modified_accuracy
from Function.Timezone2Date import Timezone2Date
from Function.TrainDataSplit import TrainDataSplit
import os
import pandas as pd

file_save_path = r"D:\Temp\StockData\US_STOCK_DATA"
stock_data_dir = os.path.join(file_save_path, r'StockData')
sort_result_dir = os.path.join(file_save_path, 'sort_result')
merged_data_dir = os.path.join(file_save_path, 'MergedData')
train_data_dir = os.path.join(file_save_path, 'TrainData')
features_data_dir = os.path.join(file_save_path, 'ProcessData')
region = 'US'
window = 2
output_directory = os.path.join(file_save_path, 'TWPredictedData')  # 儲存預測結果CSV檔案的目錄
outAllDir = os.path.join(file_save_path, 'TWPredictedData_All')


def ValidDataPreProcess():
    # Timezone2Date(stock_data_dir)
    # MergeOriginCSV(sort_result_dir, stock_data_dir, merged_data_dir, region)
    DropColumns(merged_data_dir, train_data_dir, region)
    FeaturesProcess(train_data_dir, features_data_dir, window)
    TrainDataSplit(features_data_dir, 650, features_data_dir)


def ValidDataCalAccuracy():
    merge_data(merged_data_dir, output_directory, outAllDir)
    extracted_files_correct_path = os.listdir(outAllDir)

    modified_accuracies = []
    for file_name in extracted_files_correct_path:
        file_path = os.path.join(outAllDir, file_name)
        df = pd.read_csv(file_path)
        accuracy = calculate_modified_accuracy(df)
        modified_accuracies.append(accuracy)

    # 計算總體準確度
    overall_modified_accuracy = sum(modified_accuracies) / len(modified_accuracies)
    print(f"Overall Modified Accuracy: {overall_modified_accuracy}")


if __name__ == '__main__':
    ValidDataPreProcess()
    # 要先去ubuntu conda auto-sklearn內使用模型預測出預測結果，再使用ValidDataCalAccuracy來預測模型驗證集準確度
    # ValidDataCalAccuracy()
