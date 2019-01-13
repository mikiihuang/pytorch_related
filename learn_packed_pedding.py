import numpy as np


word2id = {}
index = 1
# 记录一下每个句子的长度
lengths = []
with open("test.txt") as f:
    contents = f.readlines()
    for line in contents:
        words_line = line.strip().split(" ")
        lengths.append(len(words_line))
        for word in words_line:
            if word not in word2id:
                word2id[word] = index
                index = index+1
# print(max(lengths))
padding_matrix = np.zeros((len(lengths),max(lengths)))
# print(padding_matrix.shape)
# print(padding_matrix)
with open("test.txt") as f2:
    lines  =f2.readlines()
    id=0
    for line in lines:
        words_line = line.strip().split(" ")
        for i in range(len(words_line)):
            padding_matrix[id,i] = word2id[words_line[i]]
        id = id+1
# print(padding_matrix[0])

import torch
import torch.nn as nn
from torch.autograd import Variable

# 降序方法1
sort_index = np.argsort(-np.array(lengths))
print(sort_index)
lengths = torch.tensor(lengths)
# print(lengths)
padding_tensor = torch.tensor(padding_matrix)
# print(padding_tensor)
# 降序方法2
_, idx_sort = torch.sort(lengths, dim=0, descending=True)
print(idx_sort)
# print(lengths[idx_sort])
_, idx_unsort = torch.sort(idx_sort, dim=0)

# print(idx_unsort)
# print(torch.index_select(padding_tensor,0,idx_sort))
x = torch.index_select(padding_tensor,0,idx_sort)
# print(x)
# print(x.shape)
x_packed = nn.utils.rnn.pack_padded_sequence(input=x, lengths=lengths[idx_sort],batch_first=True)
# print(x_packed.data)
x_padded = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
print(x_padded)
output = x_padded[0].index_select(0, idx_unsort)

