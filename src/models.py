from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

class BPMModel:
    def __init__(self, model_type='linear'):
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge()
        elif model_type == 'lasso':
            self.model = Lasso()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor()
        else:
            raise ValueError("Unsupported model type. Choose from 'linear', 'ridge', 'lasso', or 'random_forest'.")

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        rmse = mean_squared_error(y, y_pred, squared=False)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return {'RMSE': rmse, 'MAE': mae, 'R²': r2}