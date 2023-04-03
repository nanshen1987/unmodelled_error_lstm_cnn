import math
import numpy as np


def computeCorrelation(x: list, y: list) -> float:
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    SSR = 0
    var_x = 0  # x的方差
    var_y = 0  # y的方差
    for xi, yi in zip(x, y):
        diff_x = xi - x_mean
        diff_y = yi - y_mean
        SSR += diff_x * diff_y
        var_x += diff_x ** 2
        var_y += diff_y ** 2
    SST = math.sqrt(var_x * var_y)
    return SSR / SST

 #决定系数
 # from sklearn.metrics import r2_score
 # r2_score(y_true,y_pred)


#均方误差、均方根误差
# from sklearn.merics import mean_squared_error
# mes = mean_squared_error(y_true,y_pred)
# rmse = np.sqrt(mse)
