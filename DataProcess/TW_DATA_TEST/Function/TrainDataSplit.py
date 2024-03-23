import os
import random
import shutil


# 獲取檔案大小的函數
def get_file_size(file_path):
    return os.path.getsize(file_path)


# 創建目錄並移動檔案的函數
def organize_files(files, base_dir, target_dir_prefix, target_size_limit):
    current_dir_path = None
    current_dir_size = 0
    current_dir_index = 1

    for file in files:
        file_size = get_file_size(file)
        if current_dir_path is None or current_dir_size + file_size > target_size_limit:
            # 如果當前目錄已滿或尚未創建，則創建新目錄
            current_dir_path = os.path.join(base_dir, f"{target_dir_prefix} {current_dir_index}")
            os.makedirs(current_dir_path, exist_ok=True)
            current_dir_index += 1
            current_dir_size = 0

        # 將檔案移動到當前目錄
        shutil.move(file, os.path.join(current_dir_path, os.path.basename(file)))
        current_dir_size += file_size


def TrainDataSplit(process_data_path, bytes_of_size, base_dir):
    """
    # 定義ProcessData目錄的路徑
    process_data_path = '/mnt/data/ProcessData'
    # 定義目標目錄和大小限制（600MB）
    base_dir = '/mnt/data'
    """
    target_size_limit = bytes_of_size * 1024 * 1024  # 600MB in bytes
    target_dir_prefix = "ProcessData"

    # 列出所有CSV檔案，並篩選掉小於20KB的檔案
    csv_files = [os.path.join(root, file) for root, dirs, files in os.walk(process_data_path) for file in files if
                 file.endswith('.csv') and get_file_size(os.path.join(root, file)) >= 20480]

    # 隨機打亂檔案順序
    random.shuffle(csv_files)

    # 執行檔案組織
    organize_files(csv_files, base_dir, target_dir_prefix, target_size_limit)
