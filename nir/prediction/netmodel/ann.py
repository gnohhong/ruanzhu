import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

class BP(nn.Module):
    def __init__(self, size, len):
        super(BP, self).__init__()
        self.fc1 = nn.Linear(size, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 4)
        self.fc4 = nn.Linear(len*4, 1)
        self.relu = nn.ReLU()
        self.lkrelu = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.lkrelu(self.fc1(x))
        x = self.lkrelu(self.fc2(x))
        x = self.lkrelu(self.fc3(x))
        x = self.fc4(x.view(-1))
        return x

class CNN(nn.Module):
    def __init__(self, size, len):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(7, 28, 8, 1, 2), #output shape (32,7,7)
                                  nn.LeakyReLU(),
                                  nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv1d(14, 56, 8, 1, 2), #output shape (32,7,7)
                                  nn.LeakyReLU(),
                                  nn.MaxPool2d(4))
        self.fc1 = nn.Linear(420, len * 5)
        self.fc2 = nn.Linear(len * 5, 1)
        self.relu = nn.ReLU()
        self.lkrelu = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.conv1(x) # 卷一次
        x = self.conv2(x) # 卷两次
        x = x.view(x.size(0), -1)
        x = self.lkrelu(self.fc1(x))
        x = self.fc2(x)
        return x

class ANNModel:
    def __init__(self, args):
        # parameters
        self.seq_len = 7
        self.input_size = 256
        self.lr = args.lr
        self.epochs = int(args.epochs)
        self.batch_size = 1
        self.cuda = args.cuda
        # The Network model
        self.model = CNN(self.input_size, self.seq_len)
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
        
    def train(self):
        self.Y = self.Y / 100
        train_loss = 0.0
        self.losslistx = np.array([])
        self.losslisty = np.array([])

        loss_function = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), self.lr)
        if self.cuda == True: # GPU训练
            loss_function = loss_function.cuda()

        print("ANN: Get train data size: " + str(len(self.X)))
        self.model.train() # 训练模式
        for i in range(self.epochs):
            for x, y in zip(self.X, self.Y):
                # n条光谱数据看做序列，作为输入
                inputs = torch.tensor(x).float().view(self.batch_size, self.seq_len, self.input_size)
                # 水分值作为单输出，作为目标值 [batch_size, seq_len, input_size]
                target = torch.tensor(y).float().view(1, self.batch_size, 1)
                if self.cuda == True: # GPU训练
                    inputs, target = inputs.cuda(), target.cuda()

                # 模型计算
                output = self.model(inputs)

                # 与上一个批次的计算图分离 https://www.cnblogs.com/catnofishing/p/13287322.html
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
            x = torch.tensor(x).float().view(self.batch_size, self.seq_len, self.input_size)
            if self.cuda == True: # GPU训练
                x = x.cuda()
            y = self.model(x)
            y_pred = np.append(y_pred, y.view(-1).cpu().item())

        return y_pred, self.Y
        
    def predict(self):
        self.model.eval() # 预测模式
        y_pred = np.array([])
        for x in self.X:
            x = torch.tensor(x).float().view(self.batch_size, self.seq_len, self.input_size)
            if self.cuda == True: # GPU训练
                x = x.cuda()
            y = self.model(x)
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