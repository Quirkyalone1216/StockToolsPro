from Timezone2Date import Timezone2Date
from MergeOriginCSV import MergeOriginCSV
from DropColumns import DropColumns
from FeaturesProcess import FeaturesProcess
from TrainDataSplit import TrainDataSplit

stock_data_dir = r'D:\Temp\StockData\US_STOCK_DATA\StockData'
sort_result_dir = r'D:\Temp\StockData\US_STOCK_DATA\sort_result'
merged_data_dir = r'D:\Temp\StockData\US_STOCK_DATA\MergedData'
train_data_dir = r'D:\Temp\StockData\US_STOCK_DATA\TrainData'
features_data_dir = r'D:\Temp\StockData\US_STOCK_DATA\ProcessData'
region = 'US'
window = 2

# Timezone2Date(stock_data_dir)
# MergeOriginCSV(sort_result_dir, stock_data_dir, merged_data_dir, region)
DropColumns(merged_data_dir, train_data_dir, region)
# FeaturesProcess(train_data_dir, features_data_dir, window)
# TrainDataSplit(features_data_dir, 650, features_data_dir)
