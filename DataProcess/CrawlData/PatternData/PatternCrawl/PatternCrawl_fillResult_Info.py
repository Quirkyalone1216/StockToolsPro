import re

tw_stock_list_path = '..\\..\\OHLC_DATA\\TW_STOCK_LIST.txt'
pattern_tw_today_path = 'Pattern_TW_Today.txt'


# 解析 TW_STOCK_LIST 檔案
def parse_stock_list(file_path):
    stock_mapping = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().split('\n')[1:]
    for line in lines:
        parts = re.split(r'\s+', line.strip())
        if len(parts) >= 5:
            stock_code = parts[1]
            stock_name = parts[2]
            industry = parts[4]
            stock_mapping[stock_code] = {'name': stock_name, 'industry': industry}
    return stock_mapping


# 使用股票名稱和產業更新 Pattern_TW_Today 檔案的函數
def update_pattern_tw_today(stock_mapping, input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        if "Stock:" in line:
            stock_code_match = re.search(r"Stock: (\d+\.TW)", line)
            if stock_code_match:
                stock_code = stock_code_match.group(1).split('.')[0]  # 移除 ".TW" 部分以進行匹配
                if stock_code in stock_mapping:
                    stock_info = stock_mapping[stock_code]
                    updated_line = f"{line.strip()} - {stock_info['name']} - {stock_info['industry']}\n"
                else:
                    updated_line = line
            else:
                updated_line = line
        else:
            updated_line = line
        updated_lines.append(updated_line)

    with open(output_path, 'w', encoding='utf-8') as file:
        file.writelines(updated_lines)


def fillResultInfo():
    stock_mapping = parse_stock_list(tw_stock_list_path)

    # 使用映射更新 Pattern_TW_Today 檔案
    output_path = r'D:\Temp\StockData\TW_STOCK_DATA\Updated_Pattern_TW_Today.txt'  # 輸出檔案路徑
    update_pattern_tw_today(stock_mapping, pattern_tw_today_path, output_path)

    print(f"已保存更新後的 Pattern_TW_Today 檔案至 {output_path}")
