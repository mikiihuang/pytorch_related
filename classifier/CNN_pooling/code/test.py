import torch
import torch.nn as nn
import numpy as np

# conv1 = nn.Conv1d(in_channels=100,out_channels=2,kernel_size=2)
# input = torch.randn(32,35,100)
# # batch_size x text_len x embedding_size -> batch_size x embedding_size x text_len
# input = torch.transpose(input,1,2)
# print(input.size())
# out = conv1(input)
# print(out.size())
#
#
# print("--------conv2d------")
#
# conv2 = nn.Conv2d(in_channels=1,out_channels=2,kernel_size=(2,100))
# input2 = torch.randn(32,35,100)
# out2 = conv2(input2)
# print(out2.size())

# # 16表示input_channel,是词嵌入的维度，33是out_channel，是用几个卷积核进行卷积，3是表示卷积核的大小，这里是（3*词嵌入的维度）
# m = nn.Conv1d(100, 2, 3, stride=1)
# # input2 = torch.randn()
# # 输入：N*C*L:batch_size为20，C为词嵌入的维度，50为句子的length
# # 输出：N*Cout*Lout：Cout我理解的是out_channel的数量
# input2 = torch.randn(20,100, 44)
# output2 = m(input2)
# print(output2.size())

# x = torch.randn(5,2, 1)
# print(x.shape)
# # print()
# out1 = torch.cat((x, x, x), 0)
# out2 = torch.cat((x, x, x), 1)
# print(out1.shape)
# print(out2.shape)
# print(torch.cat((x, x, x), 0))
# print(torch.cat((x, x, x), 1))

one = np.ones((5,3))
print(one)
print(one*0)