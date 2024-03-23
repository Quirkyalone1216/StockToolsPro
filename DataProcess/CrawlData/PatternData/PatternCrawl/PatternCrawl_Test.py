import datetime
from Function import GetKBar
from PatternCrawl_CalAllPattern import CalAllPattern
from PatternCrawl_SortResult import *


def ManMoveSetting():
    SDate = '18000101'
    EData = datetime.datetime.today().strftime('%Y%m%d')
    Prod = '0050.TW'
    Kind = 'Stock'
    Cycle = '1d'

    Data = GetKBar(SDate, EData, Prod, Kind, Cycle)
    print(Data)


def AllPatternTest():
    SDate = '18000101'
    EData = datetime.datetime.today().strftime('%Y%m%d')
    Prod = '0050.TW'
    Kind = 'Stock'
    Cycle = '1d'

    os.makedirs('result', exist_ok=True)
    CalAllPattern(SDate, EData, Prod, Kind, Cycle, 'result')


def SummarizePatternTest():
    result_path = 'result'
    output_dir = 'sort_result'
    mode = 'All'
    SummarizePattern(result_path, output_dir, mode)


def SortPatternTest():
    result_path = 'result'
    output_dir = 'sort_result'
    mode = 'All'
    sortResult(result_path, output_dir, mode)


def Crawl_60min_KBar_Pattern():
    # 獲取今天的日期
    today = datetime.date.today()

    # 使用timedelta增加一天
    time_end = (today + datetime.timedelta(days=1)).strftime("%Y%m%d")
    time_start = (today - datetime.timedelta(days=6)).strftime("%Y%m%d")

    SDate = time_start
    EData = time_end
    Prod = '0050.TW'
    Kind = 'Stock'
    Cycle = '60m'

    os.makedirs('test', exist_ok=True)
    CalAllPattern(SDate, EData, Prod, Kind, Cycle, 'test')

    result_path = 'test'
    output_dir = 'sort_test'
    outputFileName = 'Pattern_TW_Test.txt'
    mode = 'Highlight'
    SortOutput(result_path, output_dir, outputFileName, mode)


Crawl_60min_KBar_Pattern()
# SortPatternTest()
