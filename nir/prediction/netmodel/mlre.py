import numpy as np
import matplotlib.pyplot as plt
from utils.dataloader import Dataloader
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']   # 雅黑字体

# 一元线性回归
class SLR:
    def __init__(self):
        self.X = np.array([])
        self.Y = np.array([])
        self.W = 0
        self.b = 0

    def format(self, data, index, ave = 1):
        # 一元线性回归，只取向量中的一个值（默认1430nm）
        i = x = y = 0
        for intens, intime, humid in data:
            x += intens[index]
            y += humid
            if (i >= ave - 1): # 计算平均值
                self.X = np.append(self.X, x / ave)
                self.Y = np.append(self.Y, y / ave)
                i = x = y = 0
            else:
                i = i + 1
        # 使用scikit-learn中的线性回归求解
        model = LinearRegression()   # 使用模型
        model.fit(self.X.reshape(-1, 1), self.Y.reshape(-1, 1))
        self.W = model.coef_
        self.b = model.intercept_
        # 计算预测值
        _Y = []
        for eachx in self.X:
            _Y.append(self.W * eachx + self.b)
        return _Y, self.Y
    
    def showanalysis(self):
        plt.clf()
        plt.title('SLR 线性回归图')
        plt.scatter(self.X, self.Y, color='red') # 绘制散点图
        _Y = np.array([])
        for _x in self.X:
            _Y = np.append(_Y, _x * self.W + self.b)
        plt.plot(self.X, _Y.reshape(-1, 1) , color='green') # 绘制折线图
        plt.show()

    def get_W_b(self):
        return self.W, self.b


class PLSR:
    def __init__(self, index = 2, mode = 'Single'):
        # 创建PLSRegression对象，并指定主成分个数
        self.pls = PLSRegression(n_components=index)
        #self.pls = KNeighborsRegressor(n_neighbors=index)

        self.scaler = StandardScaler()

        self.X = np.array([])
        self.Y = np.array([])
        self.mode = mode

    def loaddata(self, data):
        intens, intime, humid = zip(*data)
        if self.mode == 'Single': # 单光谱输入
            for inten in intens:
                if len(self.X) == 0: # 空数组列表
                    self.X = np.hstack((self.X, np.array(inten)))
                else: # 此时数组里存在值，只能通过合并
                    self.X = np.vstack((self.X, np.array(inten)))
                self.Y = np.append(self.Y, sum(filter(None, humid)))

        elif self.mode == 'Group': # 以样本为组的多光谱输入
            if len(self.X) == 0: # 空数组列表
                self.X = np.hstack((self.X, np.array(intens).flatten())) 
            else: # 此时数组里存在值，只能通过合并
                self.X = np.vstack((self.X, np.array(intens).flatten()))
            self.Y = np.append(self.Y, sum(filter(None, humid)))

    # 训练参数
    def fitpara(self):
        print("PLSR: Get train data size: " + str(len(self.X)))
        
        # 对自变量进行标准化处理
        #scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)

        # 进行偏最小二乘回归分析
        self.pls.fit(X_scaled, self.Y)

        # 获得预测值
        y_pred = self.pls.predict(X_scaled)
        return y_pred, self.Y

    # 预测结果
    def format(self):
        print("PLSR: Get predict data size: " + str(len(self.X)))

        # 对自变量进行标准化处理
        #scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        
        # 获得预测值
        y_pred = self.pls.predict(X_scaled)
        return y_pred, self.Y
    
    def clear(self):
        self.X = np.array([])
        self.Y = np.array([])
    
    def getmodel(self):
        return self.pls
    
    def setmodel(self, model):
        self.pls = model
    