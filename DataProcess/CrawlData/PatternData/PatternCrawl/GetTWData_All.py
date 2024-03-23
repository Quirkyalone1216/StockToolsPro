from PatternCrawl_GetTWPattern import GetAllStockPattern
from PatternCrawl_SortResult import SortOutput
from PatternCrawl_fillResult_Info import fillResultInfo

if __name__ == '__main__':
    GetAllStockPattern()

    result_path = 'result'
    output_dir = 'sort_result'
    outputFileName = 'Pattern_TW_Today.txt'
    mode = 'All'
    region = 'TW'
    SortOutput(result_path, output_dir, outputFileName, mode, region)
    fillResultInfo()
