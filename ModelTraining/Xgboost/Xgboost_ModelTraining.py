import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# Loading and preprocessing data
def load_specific_stock_data(stock_file):
    stock_data = pd.read_csv(stock_file)
    stock_data['Year'] = pd.to_datetime(stock_data['Date']).dt.year
    stock_data['Month'] = pd.to_datetime(stock_data['Date']).dt.month
    stock_data['Day'] = pd.to_datetime(stock_data['Date']).dt.day
    # 處理日期格式
    stock_data = stock_data.drop('Date', axis=1)

    # 填充 NaN 值
    stock_data.ffill(inplace=True)
    stock_data.bfill(inplace=True)

    return stock_data


def preprocess_data(stock_data):
    X = stock_data.drop('Adj Close', axis=1)
    y = stock_data['Adj Close']
    return X, y


def mean_absolute_percentage_error(y_true, y_pred):
    return 100 * (abs((y_true - y_pred) / y_true)).mean()


# Training and optimizing the model with Early Stopping
def train_and_optimize_model(X_train, y_train):
    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, test_size=0.1)
    eval_set = [(X_val, y_val)]

    # Check if X_train_part or y_train_part contains NaN or infinity values
    if X_train_part.isnull().values.any() or y_train_part.isnull().values.any():
        # Fill NaN values with the median
        X_train_part = X_train_part.fillna(X_train_part.median())
        y_train_part = y_train_part.fillna(y_train_part.median())

    model = xgb.XGBRegressor(tree_method="hist", device="cuda", eval_metric="mae")
    model.fit(X_train_part, y_train_part, eval_set=eval_set, verbose=False)
    return model


# Bayesian Optimization for hyperparameter tuning
def hyperopt_tuning(X_train, y_train):
    def objective(params):
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        predictions = model.predict(X_train)
        error = mean_absolute_percentage_error(y_train, predictions)
        return {'loss': error, 'status': STATUS_OK}

    space = {
        'learning_rate': hp.loguniform('learning_rate', -5, 0),
        'max_depth': hp.choice('max_depth', [3, 4, 5, 6, 7, 8, 9, 10]),
        'n_estimators': hp.choice('n_estimators', [50, 100, 200, 300, 400, 500]),
        'tree_method': 'hist'
    }

    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=Trials())
    return best


# Plot feature importance
def plot_feature_importance(model):
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model)
    plt.show()


# Save the model
def save_model(model, model_name):
    # 检查目录是否存在
    if not os.path.exists("model"):
        # 如果不存在，创建目录
        os.makedirs("model")
    model.save_model(f"..\\Xgboost\\model\\{model_name}.model")


# Main function
def main():
    stock_files = os.listdir("..\\..\\CrawlData\\stock_data\\")
    for stock_file in stock_files:
        print(f"Training on {stock_file}...")
        stock_data = load_specific_stock_data(f'..\\..\\CrawlData\\stock_data\\{stock_file}')
        X, y = preprocess_data(stock_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = train_and_optimize_model(X_train, y_train)
        predictions = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, predictions)
        print(f"MAPE for {stock_file}: {mape}%")

        while (mape > 10):
            print("Applying Bayesian Optimization...")
            best_params = hyperopt_tuning(X_train, y_train)
            model = xgb.XGBRegressor(tree_method="hist", device="cuda", **best_params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, predictions)
            print(f"MAPE after Bayesian Optimization for {stock_file}: {mape}%")

        # plot_feature_importance(model)
        save_model(model, stock_file.split('.')[0])  # Saving the model with the stock name


if __name__ == "__main__":
    main()
