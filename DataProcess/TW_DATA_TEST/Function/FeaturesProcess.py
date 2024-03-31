import pandas as pd
import os
from ta.trend import ADXIndicator, MACD
from ta.volume import OnBalanceVolumeIndicator, VolumePriceTrendIndicator, AccDistIndexIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import RSIIndicator, StochasticOscillator


# 添加滾動和滯後特徵
def add_features(df, window_day):
    try:
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        df.sort_values('Date', inplace=True)

        for window in [5, 10, 20, 60, 120]:
            df[f'MA{window}'] = df['Close'].rolling(window=window).mean()

        # 計算移動平均線
        for window in [5, 10, 20, 60, 120]:
            df[f'MA{window}'] = df['Close'].rolling(window=window).mean()

        # 計算PVT（價格量趨勢指標）
        df['PVT'] = VolumePriceTrendIndicator(df['Close'], df['Volume']).volume_price_trend()

        # 計算OBV（能量潮指標）
        df['OBV'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

        # 計算ATR（平均真實範圍）
        atr_indicator = AverageTrueRange(df['High'], df['Low'], df['Close'])
        df['ATR'] = atr_indicator.average_true_range()

        # 計算ADX（平均方向移動指數）
        adx_indicator = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], fillna=True)
        df['ADX'] = adx_indicator.adx()

        # 新增指標
        # 計算VWAP（成交量加權平均價格）
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

        # 計算ADL（累積/分布線）
        df['ADL'] = AccDistIndexIndicator(high=df['High'], low=df['Low'], close=df['Close'],
                                          volume=df['Volume']).acc_dist_index()
        """
        # 計算RSI（相對強弱指數）
        df['RSI'] = RSIIndicator(close=df['Close']).rsi()

        # 計算隨機振盪器（Stochastic Oscillator）
        stoch_osc = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], fillna=True)
        df['Stoch_Osc'] = stoch_osc.stoch()

        # 計算布林帶（Bollinger Bands）
        bb_indicator = BollingerBands(close=df['Close'])
        df['BB_High'] = bb_indicator.bollinger_hband()
        df['BB_Low'] = bb_indicator.bollinger_lband()

        # 計算MACD（移動平均收斂/發散指標）
        macd_indicator = MACD(close=df['Close'])
        df['MACD'] = macd_indicator.macd()
        """

        # df = df.astype('float32')
        df = df.dropna()

        # 滾動特徵
        df['Future_Signal'] = df['Signal'].shift(window_day * -1)
        df = df.drop(columns=['Date', 'Year', 'Month', 'Day', 'DayOfWeek', 'DaysFromStart'])

        return df
    except Exception as e:
        print(e)


def FeaturesProcess(extraction_dir, SaveDir, window):
    """
    # 設定路徑
    extraction_dir = 'MergedTrainCSV'
    """

    extracted_files = os.listdir(extraction_dir)

    for file in extracted_files:
        print(file)
        try:
            file_path = os.path.join(extraction_dir, file)
            data = pd.read_csv(file_path)

            # 為DataFrame添加滾動和滯後特徵
            data_with_features = add_features(data, window)

            os.makedirs(SaveDir, exist_ok=True)
            SaveFilePath = os.path.join(SaveDir, file)

            data_with_features.to_csv(SaveFilePath, index=False)
        except Exception as e:
            print(e)
