import pandas as pd
import os


# 添加滾動和滯後特徵
def add_rolling_lag_features(df, window):
    df['Rolling_Mean_Close'] = df['Close'].rolling(window=window).mean()
    df['Rolling_Max_Close'] = df['Close'].rolling(window=window).max()
    df['Rolling_Min_Close'] = df['Close'].rolling(window=window).min()
    df['Rolling_Std_Close'] = df['Close'].rolling(window=window).std()

    # 滯後特徵
    for i in range(1, window + 1):
        df[f'Lag_Close_{i}'] = df['Close'].shift(i)

    # 滾動特徵
    df['Future_Signal'] = df['Signal'].shift(window * -1)

    df = df.astype('float32')
    cleaned_df = df.dropna()
    return cleaned_df


def FeaturesProcess(extraction_dir, SaveDir, window):
    """
    # 設定路徑
    extraction_dir = 'MergedTrainCSV'
    """

    extracted_files = os.listdir(extraction_dir)

    for file in extracted_files:
        file_path = os.path.join(extraction_dir, file)
        data = pd.read_csv(file_path)

        # 為DataFrame添加滾動和滯後特徵
        data_with_features = add_rolling_lag_features(data, window)

        os.makedirs(SaveDir, exist_ok=True)
        SaveFilePath = os.path.join(SaveDir, file)
        data_with_features.to_csv(SaveFilePath, index=False)
