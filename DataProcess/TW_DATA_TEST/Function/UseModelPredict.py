import pandas as pd
import os
import shutil


def preprocess_data(df):
    # 保留 'Date' 和 'Pattern' 列以便后续使用
    df_processed = df.drop(labels=['Signal'], axis=1, errors='ignore')
    return df_processed


def clear_directory(directory_path):
    # 檢查目錄是否存在
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 刪除檔案或符號鏈接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 刪除目錄
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def process_files_and_predict(directory_path, model, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    clear_directory(output_directory)

    file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.csv')]

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df = preprocess_data(
            df)  # for using model: models_signal1.joblib to predict signals, if not using models_signal1.joblib can
        # bypass this step.
        features = df.drop(['Date', 'Pattern'], axis=1, errors='ignore')  # 假设其他列是特征

        # 在进行预测前检查 features 是否为空
        if features.empty or features.shape[1] == 0:
            print(f"Warning: No features to predict for file {file_path}. Skipping.")
            continue

        # 使用模型进行预测
        try:
            prediction = model.predict(features)
            # 將0轉換為-100，將1轉換為100
            prediction_transformed = [-100 if x == 0 else 100 for x in prediction]
            df_predictions = pd.DataFrame(prediction_transformed, columns=['Predicted_Signal'])
            df['Predicted_Signal'] = df_predictions
            output_file_path = os.path.join(output_directory, os.path.basename(file_path))
            df.to_csv(output_file_path, index=False)
        except ValueError as e:
            print(f"Error predicting for file {file_path}: {e}")
            continue
