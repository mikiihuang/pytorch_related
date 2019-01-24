import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
lstm = nn.LSTM(3,3)
# 也就是seq_len是5,5个time-step
inputs = [torch.randn(1,3) for _ in range(5)]
# print(inputs)
# print(torch.randn(1,1,3))

# 也可以在类里自己定义初始化,维度是(num_layers * num_directions, batch, hidden_size)
hidden = (torch.randn(1,1,3),torch.randn(1,1,3))


# 逐个读入序列的元素
for i in inputs:
    # print(i.view(1,1,-1))
    out,hidden = lstm(i.view(1,1,-1),hidden)
    # print("out\n",out)
    # print("hidden\n",hidden)

#以下是直接读入整个序列,LSTM返回的第一个值表示所有时刻的隐状态值，第二个表示最近的隐状态值
# 所以下面的out_all和hidden_all是一样的
#out_all是最后一层每个time-step的输出值,这里我们只有一层LSTM。
# hidden_all的第一个张量h_n表示的是最后一个time_step的值

# batch_size为1，sequence_length为len(inputs)
inputs = torch.cat(inputs).view(len(inputs),1,-1)
# print(inputs)
out_all,hidden_all = lstm(inputs,hidden)
# print(out_all)
# print(hidden_all)