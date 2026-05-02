import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']   # 雅黑字体

"""
https://blog.csdn.net/great_yzl/article/details/120979935
Elman network:
    ht = oh * (Wxt + Uht-1 + bh)
    yt = oy * (Vht + by)
Jordan network:
    ht = oh * (Wxt + Uyt-1 + bh)
    yt = oy * (Vht + by)
"""

# 定义 Elman 神经网络模型
class ElmanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_len):
        super(ElmanRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity='relu',
            batch_first=False,
            bidirectional=False,
        )
        '''self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=False,
        )'''
        '''self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=False,
        )'''
        self.linear1 = nn.Linear(hidden_size, output_size)
        self.linear2 = nn.Linear(seq_len, output_size)

    def forward(self, inputs, h0):
        # x: [seq_len, batch_size, input_size]
        # h0: [num_layers, batch_size, hidden_size]
        outputs, hn = self.rnn(inputs, h0)
        # out: [seq_len, batch_size, hidden_size]
        # hn: [num_layers, batch_size, hidden_size]

        # [seq_len, batch, hidden_size] => [batch * seq_len, hidden_size]
        outputs = outputs.view(-1, self.hidden_size)
        # [batch_size * seq_len, hidden_size] => [batch_size * seq_len, output_size]
        outputs = self.linear1(outputs)

        outputs = outputs.view(-1)
        outputs = self.linear2(outputs)
        # [batch_size * seq_len, output_size] => [1, batch_size, 1]
        outputs = outputs.view(1, inputs.size(1), 1)

        return outputs, hn

class ElmanModel:
    def __init__(self, args):
        # parameters
        self.seq_len = 7
        self.input_size = 256
        self.hidden_size = 64
        # self.hidden_size = 11
        self.output_size = 1
        self.num_layers = 1
        self.lr = args.lr
        self.epochs = int(args.epochs)
        self.batch_size = 1
        self.cuda = args.cuda
        # The Network model
        self.model = ElmanRNN(self.input_size, self.hidden_size, self.num_layers, self.output_size, self.seq_len)
        if self.cuda == True: # GPU训练
            self.model.cuda()
        # utils
        self.losslistx = np.array([])
        self.losslisty = np.array([])
        # data
        self.X = np.array([]) # n * 256 * 1
        self.Y = np.array([]) # n * 1

    def loaddata(self, data):
        intens, intime, humid = zip(*data)
        if len(self.X) == 0: # 空数组列表
            self.X = np.array([intens])
        else: # 此时数组里存在值，只能通过合并
            self.X = np.append(self.X, [intens], axis=0)
        self.Y = np.append(self.Y, sum(filter(None, humid)))

    def cleardata(self):
        self.X = np.array([]) # n * 256 * 1
        self.Y = np.array([]) # n * 1
        
    def train(self):
        self.Y = self.Y / 100
        train_loss = 0.0
        self.losslistx = np.array([])
        self.losslisty = np.array([])

        loss_function = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), self.lr)
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        if self.cuda == True: # GPU训练
            loss_function, h0 = loss_function.cuda(), h0.cuda()

        print("Elman RNN: Get train data size: " + str(len(self.X)))
        self.model.train() # 训练模式
        for i in range(self.epochs):
            for x, y in zip(self.X, self.Y):
                # n条光谱数据看做序列，作为输入
                inputs = torch.tensor(x).float().view(self.seq_len, self.batch_size, self.input_size)
                # 水分值作为单输出，作为目标值 [batch_size, seq_len, input_size]
                target = torch.tensor(y).float().view(1, self.batch_size, 1)
                if self.cuda == True: # GPU训练
                    inputs, target = inputs.cuda(), target.cuda()

                # 模型计算
                output, hn = self.model(inputs, h0)

                # 与上一个批次的计算图分离 https://www.cnblogs.com/catnofishing/p/13287322.html
                hn.detach()
                loss = loss_function(output, target)
                self.model.zero_grad()
                loss.backward()
                optimizer.step()
            
                train_loss += loss.item()

            if i % 10 == 0: # 每训练10轮打印一次loss
                print("[Epoch: {}]: loss {}".format(i, train_loss))
                self.losslistx = np.append(self.losslistx, i)
                self.losslisty = np.append(self.losslisty, train_loss)
                train_loss = 0.0

        self.model.eval() # 预测模式
        y_pred = np.array([])
        for x in self.X:
            x = torch.tensor(x).float().view(self.seq_len, self.batch_size, self.input_size)
            if self.cuda == True: # GPU训练
                x = x.cuda()
            y, hn = self.model(x, h0)
            y_pred = np.append(y_pred, y.view(-1).cpu().item())

        return y_pred, self.Y
        
    def predict(self):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        if self.cuda == True: # GPU训练
            h0 = h0.cuda()

        self.model.eval() # 预测模式
        y_pred = np.array([])
        for x in self.X:
            x = torch.tensor(x).float().view(self.seq_len, self.batch_size, self.input_size)
            if self.cuda == True: # GPU训练
                x = x.cuda()
            y, hn = self.model(x, h0)
            y_pred = np.append(y_pred, y.view(-1).cpu().item())

        y_pred = y_pred * 100
        return y_pred, self.Y

    def showanalysis(self):
        plt.clf()
        plt.title('Loss 曲线图')
        plt.plot(self.losslistx, self.losslisty, color='red') # 绘制折线图
        plt.show()

    def getmodel(self):
        return self.model
