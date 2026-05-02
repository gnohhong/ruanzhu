import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

class Datashow:
    def __init__(self, _x = [], basey = [], standy = []):
        self.x = _x
        self.y = [[]]
        self.basey = basey
        self.standy = standy

    # 显示plt折现图
    def show(self):
        plt.clf()
        #plt.title('最大最小归一化', fontsize=20)
        #plt.xlabel('波长(nm)', fontsize=16)
        #plt.ylabel('反射率(%)', fontsize=16)
        for ye in self.y:
            plt.plot(self.x, ye)
        plt.show()

    # 显示标准光谱（只能以光强形式）
    def showstand(self):
        self.y.clear()
        for key, value in self.standy.items():
            # 标准光谱 list 每一项减去暗底
            self.y.append([value[i] - self.basey[i] for i in range(len(value))])
        self.show()

    # 显示数据（函数参数接收 zip (光强，积分时间)）
    def showdata(self, data):
        self.y.clear()
        for intens, intime, humid in data:
            self.y.append(intens)
        self.show()

    # 显示所有光强数据
    def showintensity(self, intensdata, mode = 'intensity'):
        self.y.clear()
        for intens, intime in intensdata:
            # 实际光强 = 当前光强 - 暗底
            _y = [intens[i] - self.basey[i] for i in range(len(intens))]
            if (mode == 'reflect'):
                for i in range(len(_y)):
                    # 反射率 = 实际光强 / 标准光强（积分时间需对应）
                    _y[i] = _y[i] / (self.standy[intime][i] - self.basey[i])
            self.y.append(_y)
        self.show()
    
    def show_result(self, pred, real):
        groups = [i for i in range(1, len(pred)+1)]
        plt.clf()
        plt.plot(groups, real, color='green', marker='*', label='real', linestyle='--') # 绘制折线图
        plt.plot(groups, pred, color='blue', marker='o', label='pred', linestyle='-') # 绘制折线图
        plt.xlabel('对比组数', fontsize=15)
        plt.ylabel('水分含量', fontsize=15)
        plt.legend(fontsize=14)
        #plt.plot(x,y,label = 'ReLU',linestyle='-',color='b')
        #plt.legend(['LeakyReLU'], fontsize=16)
        plt.show()

    # 显示预测结果与实际结果
    def show_rescompare(self, pred, real):
        plt.clf()
        plt.scatter(real, pred, color='red') # 绘制散点图
        plt.plot(real, real, color='green') # 绘制折线图
        plt.xlabel('真实值', fontsize=15)
        plt.ylabel('预测值', fontsize=15)
        plt.show()
