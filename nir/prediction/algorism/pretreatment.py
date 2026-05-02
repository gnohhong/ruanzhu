import numpy as np
import pandas as pd
import pywt
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy

class Pretreatment:
    def __init__(self, sg = False, snv = False, msc = False, dt = False, maxmin = False, mean = False,
                    vector = False, d1 = False, d2 = False, wave = False):
        self.sg = sg
        self.snv = snv
        self.msc = msc
        self.dt = dt
        self.maxmin = maxmin
        self.mean = mean
        self.vector = vector
        self.d1 = d1
        self.d2 = d2
        self.wave = wave

    def execute(self, data):
        _intens, _intime, _humid = zip(*data)

        if (self.sg == True): # Savitzky-Golay 平滑算法
            _intens = SGsmooth(_intens, 5, 2)
        if (self.snv == True): # 标准正态变量变换
            _intens = SNV(_intens)
        if (self.msc == True): # 多元散射校正
            _intens = MSC(_intens)
        if (self.dt == True): # 去趋势
            _intens = DT(_intens)
        if (self.maxmin == True): # 最大最小归一化
            _intens = MaxMin(_intens)
        if (self.mean == True): # 均值中心化
            _intens = Mean(_intens)
        if (self.vector == True): # 矢量归一化
            _intens = Vector(_intens)
        if (self.d1 == True): # 一阶差分
            _intens = D1(_intens)
        if (self.d2 == True): # 二阶差分
            _intens = D2(_intens)
        if (self.wave == True): # 小波变换
            _intens = Wave(_intens)

        data = zip(_intens, _intime, _humid)
        return data

"""
y: 代表曲线点坐标 (x,y) 中的y值数组
window_length: 窗口长度, 该值需为正奇整数。例如: 此处取值9
k值: polyorder为对窗口内的数据点进行k阶多项式拟合,
    k的值需要小于window_length。例如: 此处取值2
**mode**：确定了要应用滤波器的填充信号的扩展类型。
        (This determines the type of extension to use for 
        the padded signal to which the filter is applied. )
"""
def SGsmooth(data, window = 5, ploy = 2):
    for i in range(len(data)):
        tupledat = savgol_filter(data[i], window_length=window, polyorder=ploy)
        for j in range(len(data[i])): # list每一元素赋值
            data[i][j] = tupledat[j]
    return data

# 最大最小归一化
def MaxMin(data):
    data = deepcopy(np.array(data))
    if isinstance(data, pd.DataFrame):
            data = data.values
    mms = MinMaxScaler()
    res = mms.fit_transform(data.T)
    return list(res.T)

# 均值中心化
def Mean(data):
    data = deepcopy(np.array(data))
    temp1 = np.mean(data, axis=0)
    temp2 = np.tile(temp1, data.shape[0]).reshape(
        (data.shape[0], data.shape[1]))
    return list(data - temp2)

# 标准正态变量变换
def SNV(data):
    for i in range(len(data)):
        # 计算平均值
        _x = sum(data[i]) / len(data[i])
        # 计算均方根差
        sx = 0
        for j in range(len(data[i])):
            sx += (data[i][j] - _x) ** 2
        sx = (sx / (len(data[i]) - 1)) ** 0.5
        # 求解计算结果
        for j in range(len(data[i])):
            data[i][j] = (data[i][j] - _x) / sx
    return data

# 多元散射校正
def MSC(data):
    data = deepcopy(np.array(data))
    if isinstance(data, pd.DataFrame):
        data = data.values

    n = data.shape[0]  # 样本数量
    k = np.zeros(data.shape[0])
    b = np.zeros(data.shape[0])

    M = np.array(np.mean(data, axis=0))

    from sklearn.linear_model import LinearRegression

    for i in range(n):
        y = data[i, :]
        y = y.reshape(-1, 1)
        M = M.reshape(-1, 1)
        model = LinearRegression()
        model.fit(M, y)
        k[i] = model.coef_
        b[i] = model.intercept_

    spec_msc = np.zeros_like(data)
    for i in range(n):
        bb = np.repeat(b[i], data.shape[1])
        kk = np.repeat(k[i], data.shape[1])
        temp = (data[i, :] - bb) / kk
        spec_msc[i, :] = temp
    return list(spec_msc)

# 矢量归一化
def Vector(data):
    data = deepcopy(np.array(data))
    if isinstance(data, pd.DataFrame):
        data = data.values
    x_mean = np.mean(data, axis=1)  # 求均值
    x_means = np.tile(x_mean, data.shape[1]).reshape((data.shape[0], data.shape[1]), order='F')
    x_2 = np.power(data, 2)
    x_sum = np.sum(x_2, axis=1)
    x_sqrt = np.sqrt(x_sum)
    x_low = np.tile(x_sqrt, data.shape[1]).reshape((data.shape[0], data.shape[1]), order='F')
    return list((data - x_means) / x_low)

def DT(data):
    a = a

# 一阶差分
def D1(data):
    data = deepcopy(np.array(data))
    if isinstance(data, pd.DataFrame):
        data = data.values
    temp1 = pd.DataFrame(data)
    temp2 = temp1.diff(axis=1)
    temp3 = temp2.values
    return list(np.delete(temp3, 0, axis=1))
    #return list(temp3)

# 二阶差分
def D2(data):
    data = deepcopy(np.array(data))
    if isinstance(data, pd.DataFrame):
        data = data.values
    temp2 = (pd.DataFrame(data)).diff(axis=1)
    #temp3 = temp2 #
    temp3 = np.delete(temp2.values, 0, axis=1)
    temp4 = (pd.DataFrame(temp3)).diff(axis=1)
    spec_D2 = np.delete(temp4.values, 0, axis=1)
    return list(spec_D2)
    #return list(temp4.values)

# 小波变换
def Wave(data):
    data = deepcopy(np.array(data))
    if isinstance(data, pd.DataFrame):
        data = data.values

    def wave_(data_x):
        w = pywt.Wavelet('db8')  # 选用Daubechies8小波
        maxlev = pywt.dwt_max_level(len(data_x), w.dec_len)
        coeffs = pywt.wavedec(data_x, 'db8', level=maxlev)
        threshold = 0.04
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
        datarec = pywt.waverec(coeffs, 'db8')
        return datarec

    tmp = None
    for i in range(data.shape[0]):
        if (i == 0):
            tmp = wave_(data[i])
        else:
            tmp = np.vstack((tmp, wave_(data[i])))
    return list(tmp)
