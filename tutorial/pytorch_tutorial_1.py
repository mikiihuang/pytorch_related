import torch

# ##Tensor
# 生成未初始化的张量
x = torch.empty(5,3)
# 随机初始化的张量
x = torch.rand(5,3)
# 创建全0张量,填充值为long
x = torch.zeros(5,3,dtype = torch.long)
# 从已有数据创建张量
x = torch.tensor([5.5,3])
# 从已有张量创建张量,如果不指定数据类型,则默认与原有张量属性一致
x = x.new_ones(5,3,dtype=torch.double)
x = torch.randn_like(x,dtype=torch.float)


###Opreations
y = torch.rand(5,3)
# 加法1
print(x+y)
# 加法2
print(torch.add(x,y))
# 加法3
result = torch.empty(5,3)
torch.add(x,y,out=result)
print(result)
# 会改变原来参数值的加法
y.add_(x)
print(y)

###reshape tensor
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1,8) #-1表示自适应
print(x.size(),y.size(),z.size())


###tensor转numpy
a = torch.ones(5)
print(a)    #tensor
b = a.numpy()   #numpy类型

###numpy转tensor
import numpy as np
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)
