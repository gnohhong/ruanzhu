import numpy as np
import matplotlib.pyplot as plt

# 绘制ReLU函数
fig =  plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
x = np.linspace(-8,8)
#y = np.where(x<0,0,x) # 小于0输出0，大于0输出y
y = np.where(x<0,x*0.1,x) # 小于0输出0，大于0输出y
plt.xlim(-8,8)
plt.ylim(-3,8)
plt.title('LeakyReLU Function', fontsize=20)
 
ax = plt.gca() # 获得当前axis坐标轴对象
ax.spines['right'].set_color('none') # 去除右边界线
ax.spines['top'].set_color('none') # 去除上边界线
 
ax.xaxis.set_ticks_position('bottom') # 指定下边的边作为x轴
ax.yaxis.set_ticks_position('left') # 指定左边的边为y轴
 
ax.spines['bottom'].set_position(('data',0)) # 指定data 设置的bottom（也就是指定的x轴）绑定到y轴的0这个点上
ax.spines['left'].set_position(('data',0))  # 指定y轴绑定到x轴的0这个点上

ax.tick_params(axis='x',labelsize=14)
ax.tick_params(axis='y',labelsize=14)
 
plt.plot(x,y,label = 'ReLU',linestyle='-',color='b')
plt.legend(['LeakyReLU'], fontsize=16)
 
plt.show()