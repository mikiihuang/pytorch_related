import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

# unsqueeze:在dim=?的维度添加一个维度
x = torch.unsqueeze(torch.linspace(-1,1,200),dim=1)
y = x.pow(2)+0.2*torch.rand(x.size())

x, y = Variable(x),Variable(y)


class NN(torch.nn.Module):
    # 神经网络的累需要继承自torch.nn.Module,__init__和forward是自定义类的主要函数

    # 对神经网络的模块进行声明
    def __init__(self,input,hidden,output):
        # 用于继承父类的初始化函数
        super(NN,self).__init__()
        self.hidden = torch.nn.Linear(input,hidden)
        self.output = torch.nn.Linear(hidden,output)

    # 搭建
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x
net = NN(input=1,hidden=10,output=1)
optimzer = torch.optim.SGD(net.parameters(),lr=0.1)
criteroi = nn.MSELoss()
print(net)
# NN(
#   (hidden): Linear(in_features=1, out_features=10, bias=True)
#   (output): Linear(in_features=10, out_features=1, bias=True)
# )
plt.ion()   # 画图
plt.show()
for i in range(300):
    predict = net(x)
    loss = criteroi(predict,y)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    print("loss:"+str(loss.item()))
    if i % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), predict.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)



