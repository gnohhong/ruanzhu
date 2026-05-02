import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc
from matplotlib import pyplot as plt
from pylab import mpl

torch.set_printoptions(threshold=np.inf)

# 定义多头自注意力模块
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # 将输入向量拆分为多个头
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # 注意力加权求和
        attended_values = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # 经过线性变换和残差连接
        x = self.fc(attended_values) + x

        return x

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # 将输入向量拆分为多个头
        q = self.leakyrelu(self.query(x)).view(batch_size, seq_len, embed_dim, 1)
        k = self.leakyrelu(self.key(x)).view(batch_size, seq_len, embed_dim, 1)
        v = self.leakyrelu(self.value(x))

        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1))# / torch.sqrt(torch.tensor(embed_dim, dtype=torch.float))
        attn_weights = torch.mean(attn_weights, dim=-1)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        #print(attn_weights.size())

        # 注意力加权求和
        attended_values = attn_weights + v

        # 经过线性变换和残差连接
        x = self.leakyrelu(self.fc(attended_values)) + x

        return x

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        return x

# 定义Transformer神经网络模型
class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(Transformer, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.feedforward = FeedForward(embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.fc3 = nn.Linear(7, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.ln(x + self.attention(x))
        x = self.ln(x + self.feedforward(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x.view(-1))
        return x

class TSNet(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, seq_len):
        super(TSNet, self).__init__()
        self.attention = SelfAttention(embed_dim, num_heads)
        self.feedforward = FeedForward(embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 8)
        self.fc3 = nn.Linear(8, 1)
        self.lkrelu = nn.LeakyReLU()
        self.num_layers = seq_len
        self.batch_size = 1
        self.hidden_size = hidden_dim

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            #nonlinearity='tanh',
            batch_first=False,
            bidirectional=False,
        )
        
    def forward(self, x):
        x = self.ln(x + self.attention(x))
        x = self.ln(x + self.feedforward(x))
        h0 = torch.zeros(self.batch_size, self.num_layers, 256).cuda()
        x1, hn = self.gru(x, h0)
        x1 = x1[:, -1]
        x2 = torch.mean(x, dim=1)
        x = self.ln(x1+x2)
        x = self.lkrelu(self.fc1(x))
        x = self.lkrelu(self.fc2(x))
        x = self.fc3(x.view(-1))
        return x

class TNetModel:
    def __init__(self, args):
        # parameters
        self.seq_len = 7
        self.input_size = 256
        self.hidden_size = 64
        self.multihead = 2
        self.lr = args.lr
        self.epochs = int(args.epochs)
        self.batch_size = 1
        self.cuda = args.cuda
        # The Network model
        self.model = TSNet(self.input_size, self.multihead, self.hidden_size, self.seq_len)
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

    def sizedata(self):
        return self.X.size, self.Y.size

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
        if self.cuda == True: # GPU训练
            loss_function = loss_function.cuda()

        print("TNet: Get train data size: " + str(len(self.X)))
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
    