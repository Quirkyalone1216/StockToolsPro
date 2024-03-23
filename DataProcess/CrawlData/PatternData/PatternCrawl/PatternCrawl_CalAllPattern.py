# 載入套件
from talib.abstract import *
import sys
import Function  # Get KBar
import os


# 定義一個通用的技術指標計算函數
def calculate_indicator(indicator_name, KBar):
    # 使用 getattr 函數根據指定的函數名稱獲取函數物件
    indicator_func = getattr(sys.modules[__name__], indicator_name)
    # 呼叫指標函數並計算結果
    return indicator_func(KBar)


# 定義函數，計算所有指標
def CalAllPattern(SDate, EDate, Prod, Kind, Cycle, DirPath):
    # 指定要計算的技術指標名稱列表
    indicator_names = [
        "CDL2CROWS", "CDL3BLACKCROWS", "CDL3INSIDE", "CDL3LINESTRIKE",
        "CDL3OUTSIDE", "CDL3STARSINSOUTH", "CDL3WHITESOLDIERS", "CDLABANDONEDBABY",
        "CDLADVANCEBLOCK", "CDLBELTHOLD", "CDLBREAKAWAY", "CDLCLOSINGMARUBOZU",
        "CDLCONCEALBABYSWALL", "CDLCOUNTERATTACK", "CDLDARKCLOUDCOVER", "CDLDOJI",
        "CDLDOJISTAR", "CDLDRAGONFLYDOJI", "CDLENGULFING", "CDLEVENINGDOJISTAR",
        "CDLEVENINGSTAR", "CDLGAPSIDESIDEWHITE", "CDLGRAVESTONEDOJI", "CDLHAMMER",
        "CDLHANGINGMAN", "CDLHARAMI", "CDLHARAMICROSS", "CDLHIGHWAVE", "CDLHIKKAKE",
        "CDLHIKKAKEMOD", "CDLHOMINGPIGEON", "CDLIDENTICAL3CROWS", "CDLINNECK",
        "CDLINVERTEDHAMMER", "CDLKICKING", "CDLKICKINGBYLENGTH", "CDLLADDERBOTTOM",
        "CDLLONGLEGGEDDOJI", "CDLLONGLINE", "CDLMARUBOZU", "CDLMATCHINGLOW",
        "CDLMATHOLD", "CDLMORNINGDOJISTAR", "CDLMORNINGSTAR", "CDLONNECK",
        "CDLPIERCING", "CDLRICKSHAWMAN", "CDLRISEFALL3METHODS", "CDLSEPARATINGLINES",
        "CDLSHOOTINGSTAR", "CDLSHORTLINE", "CDLSPINNINGTOP", "CDLSTALLEDPATTERN",
        "CDLSTICKSANDWICH", "CDLTAKURI", "CDLTASUKIGAP", "CDLTHRUSTING", "CDLTRISTAR",
        "CDLUNIQUE3RIVER", "CDLUPSIDEGAP2CROWS", "CDLXSIDEGAP3METHODS"
    ]

    # 獲取KBar數據
    KBar = Function.GetKBar(SDate, EDate, Prod, Kind, Cycle)

    output_file_name = f"{Prod}.txt"
    output_file_path = os.path.join(DirPath, output_file_name)

    with open(output_file_path, 'w', encoding="utf-8") as output_file:
        # 寫入股票代碼
        output_file.write(f"{Prod}\n")

        for indicator_name in indicator_names:
            # 計算指定指標的結果
            Data = calculate_indicator(indicator_name, KBar)
            # 寫入指標名稱
            output_file.write(f"Pattern Recognition Name: {indicator_name}\n")
            # 遍歷計算結果，寫入日期和信號值
            flag = False
            for i in range(0, len(Data)):
                signal = Data[i]
                if signal != 0:
                    output_file.write(f"{KBar.index[i]} : {signal}\n")
                    flag = True

            if flag == False:
                output_file.write("期間內無觸發此型態訊號\n")

    return 1
