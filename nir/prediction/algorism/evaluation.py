import math
import numpy as np

class Evaluation:
    def __init__(self, ppmcc = False, mse = False, rmse = False, mae = False,
                        rd = False, r2 = False, rpd = False):
        self.ppmcc = ppmcc
        self.mse = mse
        self.rmse = rmse
        self.mae = mae
        self.rd = rd
        self.r2 = r2
        self.rpd = rpd

    def execute(self, pred, real):
        print("Evaluation result:")
        if (self.ppmcc == True): # 皮尔逊相关系数
            _ppmcc = PPMCC(pred, real)
            print("PPMCC: " + str(_ppmcc))
        if (self.mse == True): # 均方误差
            _mse = MSE(pred, real)
            print("MSE: " + str(_mse))
        if (self.rmse == True): # 均方根误差
            _rmse = RMSE(pred, real)
            print("RMSE: " + str(_rmse))
        if (self.mae == True): # 平均绝对误差
            _mae = MAE(pred, real)
            print("MAE: " + str(_mae))
        if (self.rd == True): # 相对误差
            _rd = RD(pred, real)
            print("RD: " + str(_rd))
        if (self.r2 == True): # 决定系数
            _r2 = R2(pred, real)
            print("R2: " + str(_r2))
        if (self.rpd == True): # 残差预测偏差
            _rpd = RPD(pred, real)
            print("RPD: " + str(_rpd))


# 皮尔逊相关系数
def PPMCC(X, Y):
    X = np.array(X).reshape(-1)
    Y = np.array(Y).reshape(-1)
    XY = X*Y
    EX = X.mean()
    EY = Y.mean()
    EX2 = (X**2).mean()
    EY2 = (Y**2).mean()
    EXY = XY.mean()
    numerator = EXY - EX*EY                                 # 分子
    denominator = math.sqrt(EX2-EX**2)*math.sqrt(EY2-EY**2) # 分母
    
    if denominator == 0:
        return 'NaN'
    ppmcc = numerator/denominator
    return ppmcc

# 均方误差
def MSE(X, Y):
    X = np.array(X).reshape(-1)
    Y = np.array(Y).reshape(-1)
    errors = X - Y
    squared_errors = np.square(errors)
    mse = np.mean(squared_errors)
    return mse

# 均方根误差
def RMSE(X, Y):
    mse = MSE(X, Y)
    rmse = math.sqrt(mse)
    return rmse

# 平均绝对误差
def MAE(X, Y):
    X = np.array(X).reshape(-1)
    Y = np.array(Y).reshape(-1)
    errors = X - Y
    abs_numbers = np.abs(errors)
    mae = np.mean(abs_numbers)
    return mae

# 相对误差
def RD(X, Y):
    X = np.array(X).reshape(-1)
    Y = np.array(Y).reshape(-1)
    errors = np.abs(X - Y)
    abs_numbers = errors / Y
    rd = np.mean(abs_numbers)
    return rd

# 决定系数
def R2(X, Y):
    X = np.array(X).reshape(-1)
    Y = np.array(Y).reshape(-1)
    SSE = np.square(X - Y)
    ymean = np.mean(Y)
    SST = np.square(Y - ymean)
    r2 = 1 - np.sum(SSE) / np.sum(SST)
    return r2

# 残差预测偏差
def RPD(X, Y):
    X = np.array(X).reshape(-1)
    Y = np.array(Y).reshape(-1)
    rrmse = RMSE(X, Y) / np.std(Y)
    rpd = 1 / np.sqrt(rrmse)
    return rpd
