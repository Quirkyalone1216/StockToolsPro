from Function.chkData import chkData
from Function.Cal_estimate_volumes import CalEstimatedVolumes
from Function.All_Estimate_Volumes import AllEstimateVolumes
from SelectStock.SelectStock_ShortTerm import ShortTerm


def main():
    weekly_path = r"D:\Temp\StockData\TW_STOCK_DATA\stock_data\Weekly_K"
    daily_path = r"D:\Temp\StockData\TW_STOCK_DATA\stock_data\Daily_K"
    fifteen_mins_path = r"D:\Temp\StockData\TW_STOCK_DATA\stock_data\15Minutes_K"

    chkData(weekly_path, 'Weekly_K')
    chkData(daily_path, 'Daily_K')
    chkData(fifteen_mins_path, '15m_K')

    fifteen_mins_new = r"D:\Temp\StockData\TW_STOCK_DATA\EstimatedVolumes_15Min_K"
    CalEstimatedVolumes(fifteen_mins_path, fifteen_mins_new)

    allEstimateVol = AllEstimateVolumes(fifteen_mins_new)

    met_shortTerm_stock = ShortTerm(daily_path, allEstimateVol)


if __name__ == '__main__':
    main()
