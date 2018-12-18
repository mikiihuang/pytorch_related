import torch.nn

# 构造类实例和查看内部变量
cell =torch.nn.RNNCell(input_size=3,hidden_size=5)
for name,param in cell.named_parameters():
    # print("{}={}".format(name,param))
    pass


# 构造LSTMCell类实例并用其搭建单向单层循环神经网络
import torch
# seq_len,batch_size = 6,2
# input_size,hidden_size = 3,5
# cell = torch.nn.LSTMCell(input_size,hidden_size)
# # **input** of shape `(batch, input_size)`
# # (seq_len,batch_size,input_size)
# inputs = torch.randn(seq_len,batch_size,input_size)
# # **h_0** of shape `(batch, hidden_size)`
# # **c_0** of shape `(batch, hidden_size)`
# h = torch.randn(batch_size,hidden_size)
# c = torch.randn(batch_size,hidden_size)
# hs = []
# for t in range(seq_len):
#     # Outputs: h_1, c_1
#     # **h_1** of shape `(batch, hidden_size)`
#     # **c_1** of shape `(batch, hidden_size)`
#     h,c = cell(inputs[t],(h,c))
#     hs.append(h)
# outputs = torch.stack(hs)
# print(outputs)


# 使用封装好的的整个神经网络，不用循环，不用考虑序列长度
num_layer = 2
seq_len,batch_size = 6,2
input_size,hidden_size = 3,5
rnn = torch.nn.GRU(input_size,hidden_size,num_layers=num_layer)
# **input** of shape `(seq_len, batch, input_size)`
# **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`
inputs = torch.randn(seq_len,batch_size,input_size)
h0 = torch.randn(num_layer,batch_size,hidden_size)

# **output** of shape `(seq_len, batch, num_directions * hidden_size)`
# **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`
outputs,hn = rnn(inputs,h0)
print(outputs)

