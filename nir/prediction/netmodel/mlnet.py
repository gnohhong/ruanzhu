import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

class LSSVR:
    def __init__(self, mode = 'Single'):
        # C值越大, 模型越复杂
        #self.lssvr = SVR(kernel='linear', C=45, gamma=0.2, epsilon=0.1)
        self.lssvr = RandomForestRegressor(n_estimators=7)

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
        print("LSSVR: Get train data size: " + str(len(self.X)))

        # 对自变量进行标准化处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        # 进行最小二乘支持向量机回归分析
        self.lssvr.fit(X_scaled, self.Y)

        # 获得预测值
        y_pred = self.lssvr.predict(X_scaled)
        return y_pred, self.Y

    # 预测结果
    def format(self):
        print("LSSVR: Get predict data size: " + str(len(self.X)))

        # 对自变量进行标准化处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # 获得预测值
        y_pred = self.lssvr.predict(X_scaled)
        return y_pred, self.Y

    def clear(self):
        self.X.clear()
        self.Y.clear()
    
    def getmodel(self):
        return self.lssvr
    
    def setmodel(self, model):
        self.lssvr = model

class ELM:
    def __init__(self, hiddenNodeNum, activationFunc="relu", mode='Single'):
        self.X = np.array([])
        self.Y = np.array([])
        self.mode = mode
        # beta矩阵
        self.beta = None
        # 偏置矩阵
        self.b = None
        # 权重矩阵
        self.W = None
        # 隐含层节点个数
        self.hiddenNodeNum = hiddenNodeNum
        # 激活函数
        self.activationFunc = self.chooseActivationFunc(activationFunc)

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

    def fitpara(self):
        print("ELM: Get train data size: " + str(len(self.X)))

        try:
            if self.Y.shape[1] > 1:
                raise ValueError("回归问题的输出维度必须为1")
        except IndexError:
            # 如果数据是一个array，则转换为列向量
            self.Y = np.array(self.Y).reshape(-1, 1)

        # 对自变量进行标准化处理
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        # 输入维度d，输出维度m，样本个数N，隐含层节点个数hiddenNodeNum
        n, d = self.X.shape
        # 权重系数矩阵 d*hiddenNodeNum
        self.W = np.random.uniform(-1.0, 1.0, size=(d, self.hiddenNodeNum))
        # 偏置系数矩阵 n*hiddenNodeNum
        self.b = np.random.uniform(-0.4, 0.4, size=(n, self.hiddenNodeNum))
        # 隐含层输出矩阵 n*hiddenNodeNum
        H = self.activationFunc(np.dot(self.X, self.W) + self.b)
        # 输出权重系数 hiddenNodeNum*m，β的计算公式为：((H.T*H)^-1)*H.T*T
        self.beta = np.dot(np.dot(pinv(np.dot(H.T, H)), H.T), self.Y)

    def chooseActivationFunc(self, activationFunc):
        """选择激活函数，这里返回的值是函数名"""
        if activationFunc == "sigmoid":
            return self._sigmoid
        elif activationFunc == "relu":
            return self._relu
        elif activationFunc == "sin":
            return self._sine
        elif activationFunc == "cos":
            return self._cos

    def format(self):
        print("ELM: Get predict data size: " + str(len(self.X)))
        # 样本个数
        sampleCNT = len(self.X)
        # 由于训练样本个数为b矩阵的行，用len函数获取，进行预测的时候必须满足该条件，否则下面公式索引会超出范围
        if sampleCNT > len(self.b):
            raise ValueError("训练集样本数必须大于测试机样本数")
        h = self.activationFunc(np.dot(self.X, self.W) + self.b[:sampleCNT, :])
        res = np.dot(h, self.beta)
        return res, self.Y

    def clear(self):
        self.X = np.array([])
        self.Y = np.array([])

    @staticmethod
    def score(y_true, y_pred):
        # 根据输出标签相等的个数计算得分
        if len(y_pred) != len(y_true):
            raise ValueError("维度不相等")
        totalNum = len(y_pred)
        rightNum = np.sum([1 if p == t else 0 for p, t in zip(y_pred, y_true)])
        return rightNum / totalNum

    @staticmethod
    def RMSE(y_pred, y_true):
        # Root Mean Square Error    均方根误差
        # 这里计算平均均方根误差
        # 计算公式参考：https://blog.csdn.net/yql_617540298/article/details/104212354
        try:
            if y_pred.shape[1] == 1:
                y_pred = y_pred.reshape(-1)
        except IndexError:
            pass
        
        return np.sqrt(np.sum(np.square(y_pred - y_true)) / len(y_pred))

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1 + np.exp(-x))
    
    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _sine(x):
        return np.sin(x)

    @staticmethod
    def _cos(x):
        return np.cos(x)
