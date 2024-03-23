import numpy as np
import pandas as pd
import optuna
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import talib
import os


# 1. Data Preprocessing
def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)
    data.ffill(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    return data


def add_features(data):
    # Moving Averages
    data['MA5'] = talib.SMA(data['Close'], timeperiod=5)
    data['MA20'] = talib.SMA(data['Close'], timeperiod=20)

    # RSI
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)

    # MACD
    data['MACD'], data['MACDSignal'], data['MACDHist'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26,
                                                                    signalperiod=9)

    # Bollinger Bands
    data['UpperBand'], data['MiddleBand'], data['LowerBand'] = talib.BBANDS(data['Close'], timeperiod=20)

    # Volume (simply use the existing 'Volume' feature if it exists in your data)
    # data['Volume'] = data['Volume']

    return data.dropna()


def create_sequences(data, input_days, target_days):
    input_sequences = []
    target_sequences = []
    for i in range(len(data) - input_days - target_days + 1):
        input_sequences.append(data[i:i + input_days])
        # Assuming 'Close' values are in the first column, adjust as needed
        target_sequences.append(data[i + input_days:i + input_days + target_days, 0])
    input_sequences, target_sequences = np.array(input_sequences), np.array(target_sequences)

    # Print shapes for debugging
    # print(f"Input sequences shape: {input_sequences.shape}")
    # print(f"Target sequences shape: {target_sequences.shape}")

    return input_sequences, target_sequences


def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data), scaler


# 2. Model Building and Tuning
def build_model(trial, input_shape, target_days):
    num_layers = trial.suggest_int('num_layers', 2, 20)
    units = trial.suggest_int('units', 32, 512)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2)
    weight_decay = trial.suggest_float('weight_decay', 1e-10, 1e-3)

    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    for _ in range(num_layers):
        model.add(layers.Dense(units, kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.LSTM(50, return_sequences=False))
    model.add(layers.Dense(target_days))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mse')
    # model.summary()
    return model


def objective(trial, input_sequences, target_sequences, input_shape, target_days):
    model = build_model(trial, input_shape, target_days)
    history = model.fit(input_sequences, target_sequences, epochs=50, validation_split=0.2, verbose=0)
    return min(history.history['val_loss'])


def train_best_model(best_params, input_sequences, target_sequences, input_shape, target_days):
    # Rebuild the model with the best parameters
    model = build_model(optuna.trial.FixedTrial(best_params), input_shape, target_days)
    history = model.fit(input_sequences, target_sequences, epochs=200, validation_split=0.2,
                        verbose=1)  # Train longer 200 times
    return model, history


def evaluate_and_predict(model, input_sequences, target_sequences, data, target_days, scaler):
    # Evaluate the model
    loss = model.evaluate(input_sequences, target_sequences, verbose=0)
    print("Model MSE Loss: ", loss)

    # Predict the future prices
    last_sequence = input_sequences[-1][np.newaxis, ...]  # Get the last sequence
    predicted_future = model.predict(last_sequence)[0]  # Predict next 'target_days' days

    # Inverse transform the predicted values
    predicted_future_inverted = inverse_transform(predicted_future, data[['Close', 'MA5', 'MA20', 'RSI', 'MACD',
                                                                          'MACDSignal', 'MACDHist', 'UpperBand',
                                                                          'MiddleBand', 'LowerBand', 'Volume']].values,
                                                  scaler)

    # Backtest against the last 'target_days' days
    actual_last = data['Close'].values[-target_days:]

    # # Print results
    # print("Predicted future prices (inverted): ", predicted_future_inverted)
    # print("Actual future prices: ", actual_last)

    # Calculating MAE, MSE and MAPE
    mae = np.mean(np.abs(predicted_future_inverted - actual_last))
    mse = np.mean(np.square(predicted_future_inverted - actual_last))
    mape = np.mean(np.abs((predicted_future_inverted - actual_last) / actual_last)) * 100

    # Print evaluation metrics
    print("MAE: ", mae)
    print("MSE: ", mse)
    print("MAPE: ", mape, "%")

    # Return for further analysis
    return mae, mse, mape


def inverse_transform(predictions, original_data, scaler):
    dummy = np.zeros((len(predictions), original_data.shape[1]))
    dummy[:, 0] = predictions
    inverted = scaler.inverse_transform(dummy)[:, 0]
    return inverted


def main():
    global best_model
    filepath = "..\\..\\CrawlData\\stock_data-1\\"  # please adjust the path accordingly
    input_days = 30
    target_days = 20
    absFilesPath = os.path.abspath(filepath)
    for file in os.listdir(absFilesPath):
        print(file)

        csvFile = os.path.join(absFilesPath, file)

        # 1. Data Preprocessing
        data = load_and_clean_data(csvFile)
        data_with_features = add_features(data)
        scaled_data, scaler = scale_data(data_with_features[['Close', 'MA5', 'MA20', 'RSI', 'MACD', 'MACDSignal',
                                                             'MACDHist', 'UpperBand', 'MiddleBand', 'LowerBand',
                                                             'Volume']])
        input_sequences, target_sequences = create_sequences(scaled_data, input_days, target_days)
        # Print shapes for debugging
        # print(f"Input sequences shape: {input_sequences.shape}")
        # print(f"Target sequences shape: {target_sequences.shape}")

        input_shape = (input_days, scaled_data.shape[1])

        mape = 100  # Initialize MAPE to a high value

        while mape > 6:
            # 2. Model Building and Tuning
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, input_sequences, target_sequences, input_shape, target_days),
                           n_trials=400)  # trials 400 times

            # Retrieve best parameters
            best_params = study.best_params

            # Train the best model
            best_model, history = train_best_model(best_params, input_sequences, target_sequences, input_shape, target_days)

            # Evaluate and Predict
            _, _, mape = evaluate_and_predict(best_model, input_sequences, target_sequences,
                                                                 data_with_features, target_days, scaler)

        if mape < 6:
            # Save the model
            best_model.save(f"Model/{file}")
            os.remove(csvFile)

        # best_model = keras.models.load_model(f"Model/{file}")


if __name__ == '__main__':
    main()
