import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class CustomLinearRegression:

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = None
        self.intercept = 0.0
        self.yhat = None
        self.r2 = None
        self.rsme = None

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.insert(X, 0, 1, 1)
        beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
        if self.fit_intercept:
            self.intercept = beta[0]
            self.coefficient = beta[1:]
        else:
            self.coefficient = beta

    def predict(self, X):
        if self.intercept == 0.0:
            self.yhat = X @ self.coefficient
        else:
            self.yhat = np.insert(X, 0, 1, 1) @ np.insert(self.coefficient, 0, self.intercept, 0)

    def r2_score(self, y):
        return 1 - sum((y - self.yhat) ** 2) / sum((y - y.mean()) ** 2)

    def rmse(self, y):
        return (sum((y - self.yhat) ** 2) / len(y)) ** 0.5

    def self(self):
        return {
            'Intercept': self.intercept,
            'Coefficient': self.coefficient,
            'R2': self.r2_score(y),
            'RMSE': self.rmse(y)
        }


f1 = [2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7.87]
f2 = [65.2, 78.9, 61.1, 45.8, 54.2, 58.7, 96.1, 100.0, 85.9, 94.3]
f3 = [15.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 15.2]
y = [24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 27.1, 16.5, 18.9, 15.0]
X = np.c_[f1, f2, f3]
y = np.array(y)

# custom model
clr = CustomLinearRegression(fit_intercept=True)
clr.fit(X, y)
clr.predict(X)
clr.r2_score(y)
clr.rmse(y)
# print(clr.self())

# sklearn model
model = LinearRegression(fit_intercept=True)
model.fit(X, y)
y_pred = model.predict(X)
rmse = math.sqrt(mean_squared_error(y, y_pred))
r2s = r2_score(y, y_pred)
diff = {'Intercept': clr.intercept - model.intercept_,
        'Coefficient': clr.coefficient - model.coef_,
        'R2': clr.r2_score(y) - r2s,
        'RMSE': clr.rmse(y) - rmse}
print(diff)

