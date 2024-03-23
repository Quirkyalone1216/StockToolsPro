import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import functools
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM


def load_data(filepath):
    return pd.read_csv(filepath, index_col='Date', parse_dates=True)


def add_technical_indicators(data):
    # MACD
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Moving Averages
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()

    return data


def preprocess_data(data, features_list):
    features = data[features_list].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)
    return features_scaled, scaler


def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


def build_model(input_shape, lstm_units, optimizer, modelPath):
    if os.path.exists(modelPath):
        model = load_model(modelPath)
    else:
        model = Sequential()
        model.add(LSTM(lstm_units, input_shape=input_shape))
        model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


def objective(trial, x_train, y_train, x_val, y_val, modelPath):
    lstm_units = trial.suggest_int('lstm_units', 30, 100)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])

    model = build_model((x_train.shape[1], x_train.shape[2]), lstm_units, optimizer, modelPath)
    model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=0, validation_split=0.1)

    # Evaluate the model with the eval dataset.
    score = model.evaluate(x_val, y_val, verbose=0)
    return score


def plot_results(data, train_predict_unscaled, test_predict_unscaled, look_back):
    plt.figure(figsize=(15, 8))
    plt.plot(data.index, data['Close'], label="Original data")
    plt.plot(data.index[look_back + 1:len(train_predict_unscaled) + look_back + 1], train_predict_unscaled[:, 0],
             label="Training predictions")
    plt.plot(data.index[len(train_predict_unscaled) + look_back + 1 - 1:len(data) - 1], test_predict_unscaled[:, 0],
             label="Testing predictions")
    plt.legend()
    plt.title("Stock price prediction using LSTM")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.show()


def predict_future(model, x_last, look_back, n_steps, scaler, features_list):
    future_preds = []
    x_last = x_last[np.newaxis, :, :]  # Add batch dimension

    for _ in range(n_steps):
        # Predict next step
        pred = model.predict(x_last)

        # Store the prediction
        future_preds.append(pred[0, 0])

        # Create new input data
        new_step = np.zeros((1, 1, x_last.shape[2]))
        new_step[0, 0, 0] = pred

        # Append other features (here we just repeat the last available real data)
        new_step[0, 0, 1:] = x_last[0, -1, 1:]

        # Append the new step to the input data
        x_last = np.concatenate((x_last, new_step), axis=1)

        # Keep only the most recent [look_back] steps
        x_last = x_last[:, -look_back:, :]

    # Rescale predictions
    preds_unscaled = scaler.inverse_transform(
        np.hstack((np.array(future_preds).reshape(-1, 1), np.tile(x_last[0, -1, 1:], (n_steps, 1))))
    )
    return preds_unscaled[:, 0]


# Evaluate model predictions
def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'Mean Squared Error: {mse:.2f}')
    return mae, mse


def main():
    # Load data
    filepath = "..\\..\\CrawlData\\stock_data\\"  # please adjust the path accordingly
    absFilesPath = os.path.abspath(filepath)
    for file in os.listdir(absFilesPath):
        csvFile = os.path.join(absFilesPath, file)
        data = load_data(csvFile)

        # Add technical indicators
        data = add_technical_indicators(data).dropna()

        # Define features and scale data
        features_list = ['Close', 'MACD', 'Signal_Line', 'RSI', 'MA5', 'MA10']
        features_scaled, scaler = preprocess_data(data, features_list)

        # Create dataset
        look_back = 10
        X, Y = create_dataset(features_scaled, look_back)

        # Split into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=False)

        # Build and train the LSTM model
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
        study = optuna.create_study(direction='minimize')
        study.optimize(functools.partial(objective, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                                         modelPath=f'Model\\{file}'), n_trials=3, n_jobs=5)

        # Results
        print('Number of finished trials: ', len(study.trials))
        print('Best trial:')
        trial = study.best_trial

        print('Value: ', trial.value)
        print('Params: ')
        for key, value in trial.params.items():
            print(f'    {key}: {value}')

        # 使用最優超參數重新訓練模型
        best_params = trial.params
        model = build_model((x_train.shape[1], x_train.shape[2]), best_params['lstm_units'], best_params['optimizer'],
                            f'Model\\{file}')
        model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=1, validation_split=0.1)

        """
        # Predictions
        train_predict = model.predict(x_train)
        test_predict = model.predict(x_test)

        # print(test_predict.shape)
        # print(x_test[:,-1,-5:].shape)

        # Invert predictions back to original scale
        train_predict_unscaled = scaler.inverse_transform(np.hstack((train_predict, x_train[:, -1, -5:])))
        test_predict_unscaled = scaler.inverse_transform(np.hstack((test_predict, x_test[:, -1, -5:])))


        # Plot original series vs predicted
        plot_results(data, train_predict_unscaled, test_predict_unscaled, look_back)

        """

        # Predict future prices
        x_last = x_test[-1, :, :]
        future_preds = predict_future(model, x_last, look_back, n_steps=30, scaler=scaler, features_list=features_list)
        print(future_preds)

        testData = load_data(csvFile)
        testData = testData.ffill()
        true_values = testData['Close'].iloc[-30:].values
        evaluate_predictions(true_values, future_preds)

        model.save(f'Model\\{file}')


if __name__ == "__main__":
    main()
