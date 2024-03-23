import pandas as pd
import os

def calculate_modified_accuracy(df):
    # 考慮預測的正負號
    correct_predictions = ((df['Signal_x'] * df['Signal_y']) > 0).sum()
    # 計算預測中實際或預測信號不為零
    total_predictions = ((df['Signal_x'] != 0) & (df['Signal_y'] != 0)).sum()
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions if total_predictions else 0
    return accuracy


