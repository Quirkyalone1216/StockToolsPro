import pandas as pd
import os
from collections import defaultdict


# 解析文件內容
def parse_file_content_robust(content, mode):
    data = []
    pattern_name = None
    for line in content.split('\n'):
        if 'Pattern Recognition Name:' in line:
            pattern_name = line.split(': ')[1]
        elif pattern_name and ':' in line:
            parts = line.split(' : ')
            if len(parts) == 2:
                date_str, signal_value = parts
                if mode == 'All':
                    data.append({
                        'Date': pd.to_datetime(date_str.split()[0]),
                        'Pattern': pattern_name,
                        'Signal': int(signal_value)
                    })
                elif mode == 'Highlight':
                    date_str, time_str = date_str.split()[0], date_str.split()[1]
                    data.append({
                        'Date': pd.to_datetime(date_str),
                        'Time': time_str,
                        'Pattern': pattern_name,
                        'Signal': int(signal_value)
                    })
    return data


# 處理每個文件，將其內容排序並保存到 CSV 文件中的函數
def process_and_save_file(file_path, output_dir, mode, region):
    # 從文件名中提取股票代碼
    stock_code = os.path.basename(file_path).split('.')[0]
    if region == 'TW':
        output_file_path = os.path.join(output_dir, f"{stock_code}.TW.csv")
    else:
        output_file_path = os.path.join(output_dir, f"{stock_code}.TW.csv")

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 解析內容
    parsed_data = parse_file_content_robust(content, mode)

    # 轉換為 DataFrame，排序並保存到 CSV
    if parsed_data:
        df = pd.DataFrame(parsed_data)
        if mode == 'All':
            df_sorted = df.sort_values(by='Date')
            df_sorted.to_csv(output_file_path, index=False)
        elif mode == 'Highlight':
            df_sorted = df.sort_values(by=['Date', 'Time'])
            df_sorted.to_csv(output_file_path, index=False)

    else:
        if mode == 'All':
            pd.DataFrame(columns=['Date', 'Pattern', 'Signal']).to_csv(output_file_path, index=False)
        elif mode == 'Highlight':
            pd.DataFrame(columns=['Date', 'Time', 'Pattern', 'Signal']).to_csv(output_file_path, index=False)


def sortResult(result_path, output_dir, mode, region):
    # 定義提取文件的路徑和輸出目錄

    os.makedirs(output_dir, exist_ok=True)

    # 處理“result”目錄中的每個文件
    for file_name in os.listdir(result_path):
        file_path = os.path.join(result_path, file_name)
        process_and_save_file(file_path, output_dir, mode, region)

    print("All files processed and CSVs saved.")


def SummarizePattern(output_dir, outputFileName, mode):
    # 定義型態解釋
    pattern_explanations = {
        "CDLENGULFING": "吞噬模式 - 一個重要的反轉模式，由一小根蠟燭被一大根相反顏色的蠟燭完全覆蓋形成。看跌吞噬表示賣方控制，看漲吞噬表示買方控制。",
        "CDLGAPSIDESIDEWHITE": "跳空並列白線 - 出現在上升趨勢中，是持續上升的信號，由兩根大的白色蠟燭組成，第二根蠟燭開盤價高於第一根的收盤價，但並未超過其最高價。",
        "CDLLONGLINE": "長線 - 表示該時間段內買賣雙方活動較為激烈，並且以較大的價格變動結束。長的白色蠟燭表示買方壓力，長的黑色蠟燭表示賣方壓力。",
        "CDLMARUBOZU": "光頭光腳/實體大 - 表示開盤價即為最低價，收盤價即為最高價（對於看漲光頭光腳），或開盤價即為最高價，收盤價即為最低價（對於看跌光頭光腳），顯示了非常強的一方力量。",
        "CDLCLOSINGMARUBOZU": "收盤光頭光腳 - 類似於光頭光腳，但專指收盤價極為強勢或弱勢的情況，是一方力量的明顯展現。",
        "CDLBELTHOLD": "帶抱 - 一種小型的看漲或看跌反轉模式，由一根具有小影線的長實體蠟燭組成，看漲帶抱的實體底部接近最低價，而看跌帶抱的實體頂部接近最高價。",
        "CDLSPINNINGTOP": "陀螺 - 表示市場不確定性，具有小實體和較長的上下影線，顯示買賣雙方均無法取得明顯優勢。",
        "CDL2CROWS": "二烏鴉，一個看跌反轉模式，出現在上升趨勢的頂部。",
        "CDL3BLACKCROWS": "三黑鴉，表示市場的看跌情緒，通常出現在價格下跌的開始。",
        "CDL3INSIDE": "三內部上升和下降，是一種反轉模式，取決於其出現的位置和之後的行動。",
        "CDL3LINESTRIKE": "三線擊打，一個看跌或看漲的反轉模式，具體取決於其出現的背景。",
        "CDL3OUTSIDE": "三外部上升和下降，指一個趨勢的延續或反轉。",
        "CDL3STARSINSOUTH": "南方三星，一種罕見的看跌反轉模式。",
        "CDL3WHITESOLDIERS": "三白兵，一個看漲反轉模式，出現在下跌趨勢的底部。",
        "CDLABANDONEDBABY": "棄嬰，看漲或看跌的反轉模式，具體取決於其出現的背景。",
        "CDLADVANCEBLOCK": "推進塊，一個看跌的頂部反轉模式。",
        "CDLCOUNTERATTACK": "反擊線，看跌或看漲的反轉模式。",
        "CDLDARKCLOUDCOVER": "烏雲蓋頂，一個看跌反轉模式。",
        "CDLDOJI": "十字星，代表市場猶豫不決。",
        "CDLDOJISTAR": "十字星，是潛在的反轉信號。",
        "CDLDRAGONFLYDOJI": "蜻蜓十字/蜻蜓Doji，看漲的反轉信號。",
        "CDLEVENINGDOJISTAR": "黃昏之星，一個看跌反轉模式。",
        "CDLEVENINGSTAR": "暮星，另一種看跌反轉模式。",
        "CDLGRAVESTONEDOJI": "墓碑Doji，看跌反轉信號。",
        "CDLHAMMER": "錘子，看漲反轉模式。",
        "CDLHANGINGMAN": "吊人，看跌反轉模式。",
        "CDLHARAMI": "懷孕，看漲或看跌的反轉模式。",
        "CDLHARAMICROSS": "十字懷孕，潛在的反轉信號。",
        "CDLHIKKAKE": "欺詐模式，反轉或趨勢延續的信號。",
        "CDLHOMINGPIGEON": "歸鴿，看漲的持續模式。",
        "CDLIDENTICAL3CROWS": "同形三烏鴉，一個強烈的看跌反轉模式。",
        "CDLINNECK": "內頸線，看跌的反轉模式。",
        "CDLINVERTEDHAMMER": "倒錘子，看漲反轉信號。",
        "CDLKICKING": "踢腳，看漲或看跌的反轉模式。",
        "CDLKICKINGBYLENGTH": "根據長度踢腳，強烈的反轉模式。",
        "CDLLADDERBOTTOM": "梯底，看漲的底部反轉模式。",
        "CDLLONGLEGGEDDOJI": "長腿Doji，市場不確定性的象徵。",
        "CDLMATCHINGLOW": "相同低點，看漲反轉模式。",
        "CDLMORNINGDOJISTAR": "晨星Doji，看漲反轉模式。",
        "CDLMORNINGSTAR": "晨星，另一看漲反轉模式。",
        "CDLONNECK": "外頸線，看跌的反轉模式。",
        "CDLPIERCING": "刺透模式，看漲反轉模式。",
        "CDLRICKSHAWMAN": "人力車夫Doji，市場不確定性的象徵。",
        "CDLSEPARATINGLINES": "分離線，趨勢延續的信號。",
        "CDLSHOOTINGSTAR": "流星，看跌反轉模式。",
        "CDLSHORTLINE": "短線，輕微的方向性移動。",
        "CDLSTALLEDPATTERN": "停滯模式，看跌的頂部反轉模式。",
        "CDLSTICKSANDWICH": "棒子三明治，看漲反轉模式。",
        "CDLTAKURI": "探水竿，看漲反轉信號。",
        "CDLTASUKIGAP": "透空隙，趨勢延續的信號。",
        "CDLTHRUSTING": "插入線，看跌反轉的可能性。",
        "CDLTRISTAR": "三星，看漲或看跌的反轉模式。",
        "CDLUNIQUE3RIVER": "獨特三河床，看漲的底部反轉模式。",
        "CDLUPSIDEGAP2CROWS": "上升跳空二烏鴉，看跌反轉模式。",
        "CDLXSIDEGAP3METHODS": "交叉跳空三方法，趨勢延續的信號。"
    }

    # 指定文件夾路徑
    folder_path = output_dir

    # 獲取文件夾內所有文件
    files = os.listdir(folder_path)

    # 初始化字典以儲存每個型態的股票
    pattern_groups = defaultdict(list)

    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)

        if not df.empty:
            # 假設每個文件的最後一行是最新的數據
            last_row = df.iloc[-1]
            pattern = last_row['Pattern']
            signal = last_row['Signal']  # 獲取信號值
            stock_code = file.replace('.csv', '')
            if mode == 'All':
                pattern_groups[pattern].append({"stock": stock_code, "signal": signal})
            elif mode == 'Highlight':
                # 現在包括時間資訊
                time = last_row['Time']  # 新增時間資訊
                # 包括時間資訊在內的字典
                pattern_groups[pattern].append({"stock": stock_code, "signal": signal, "time": time})

    # 將型態分組並映射到它們的解釋
    pattern_groups_with_explanations = {}
    for pattern, stock_info in pattern_groups.items():
        explanation = pattern_explanations.get(pattern, "No explanation available")
        pattern_groups_with_explanations[pattern] = {"stocks_info": stock_info, "explanation": explanation}

    output_result = []
    # 輸出結果
    for pattern, info in pattern_groups_with_explanations.items():
        # print(f"Pattern: {pattern}, Explanation: {info['explanation']}")
        output_result.append(f"Pattern: {pattern}, Explanation: {info['explanation']}")
        for stock in info["stocks_info"]:
            # print(f"Stock: {stock['stock']}, Signal: {stock['signal']}")
            if mode == 'All':
                output_result.append(f"Stock: {stock['stock']}, Signal: {stock['signal']}")
            elif mode == 'Highlight':
                output_result.append(f"Stock: {stock['stock']}, Signal: {stock['signal']}, Time: {stock['time']}")

        # print("\n")
        output_result.append('\n')

    with open(outputFileName, 'w', encoding='utf-8') as f:
        for _ in output_result:
            f.write(_)
            f.write('\n')

    # print(output_result)


def SortOutput(result_path, output_dir, outputFileName, mode, region):
    sortResult(result_path, output_dir, mode, region)
    SummarizePattern(output_dir, outputFileName, mode)
