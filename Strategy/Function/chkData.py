import os
import pandas as pd


def chkData(Path, type):
    delFiles = []
    files = os.listdir(Path)
    for file in files:
        file_path = os.path.join(Path, file)
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if len(lines) <= 1:
                    delFiles.append(file_path)
        except UnicodeDecodeError:
            pass

    for delFile in delFiles:
        os.remove(delFile)

    files = os.listdir(Path)  # 更新檔案列表

    for file in files:
        file_path = os.path.join(Path, file)
        df = pd.read_csv(file_path)

        # 檢查是否為 '15m_K' 類型且檢查 DataFrame 中的欄位
        if type == '15m_K':
            if 'Datetime' in df.columns:
                # 轉換 'Datetime' 欄位格式並重命名為 'Date'
                df['Datetime'] = pd.to_datetime(df['Datetime']).apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
                df.rename(columns={'Datetime': 'Date'}, inplace=True)
            elif 'Date' in df.columns:
                # 如果有 'Date' 但沒有 'Datetime' 欄位，則略過此步驟
                pass
        else:
            # 對於其他類型，只處理 'Date' 欄位
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(None).dt.date

        # 將修改後的 DataFrame 保存回同一檔案
        df.to_csv(file_path, index=False)
