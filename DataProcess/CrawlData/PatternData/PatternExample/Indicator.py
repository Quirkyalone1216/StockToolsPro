# 載入套件
from talib.abstract import *
import numpy as np


def Signal_01(KBar):
    Signal = [0]
    FastMA = MA(KBar, timeperiod=5, matype=0)
    SlowMA = MA(KBar, timeperiod=10, matype=0)
    for i in range(1, len(SlowMA)):
        LastFastMA = FastMA[i - 1]
        LastSlowMA = SlowMA[i - 1]
        ThisFastMA = FastMA[i]
        ThisSlowMA = SlowMA[i]
        # 避免遇到空值
        if np.isnan(LastFastMA) == False and np.isnan(LastSlowMA) == False:
            # 多方訊號 (均線黃金交叉)
            if LastFastMA <= LastSlowMA and ThisFastMA > ThisSlowMA:
                Signal.append(100)
            # 空方訊號 (均線死亡交叉)
            elif LastFastMA >= LastSlowMA and ThisFastMA < ThisSlowMA:
                Signal.append(-100)
            # 無訊號
            else:
                Signal.append(0)
        else:
            Signal.append(0)
    KBar['Signal_01'] = Signal
    return KBar


def Signal_02(KBar):
    KBar['Signal_02'] = CDL2CROWS(KBar)
    return KBar


def Signal_03(KBar):
    Signal = []
    RSI_array = RSI(KBar, timeperiod=6)
    for i in range(0, len(RSI_array)):
        R = RSI_array[i]
        # 避免遇到空值
        if np.isnan(R) == False:
            # 多方訊號 (RSI指標>=60)
            if R >= 55:
                Signal.append(100)
            # 空方訊號 (RSI指標<=40)
            elif R <= 45:
                Signal.append(-100)
            # 無訊號
            else:
                Signal.append(0)
        else:
            Signal.append(0)
    KBar['Signal_03'] = Signal
    return KBar


# 讀者可自行新增訊號
def Signal_04(KBar):
    pass
