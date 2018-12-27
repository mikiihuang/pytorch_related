
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


words = []
with open("test.txt") as f:
    cotents = f.readlines()
    # print(len(cotents))
    # all_lines = np.array(len(cotents),)
    every_line_length = []
    every_line = []
    for line in cotents:

        # print(type(line))
        line = line.replace(".","").replace(",","").replace(";","").replace("\n","")
        line_token = line.split(" ")
        every_line_length.append(len(line_token))
        every_line.append(line_token)
        for word in line_token:
            words.append(word)
        # print(words)
    largest_length = max(every_line_length)
    inputs = np.zeros((len(cotents),largest_length))
    # print(all_lines)
words_list = list(set(words))
# print(every_line)
words2id = {word:id for id,word in enumerate(words_list)}
# print(words2id)
id2word = {id:word for id,word in enumerate(words_list)}

# 把输入的文字全部转换成one-hot向量
for i in range(len(every_line)):
    for j in range(len(every_line[i])):
        # print(every_line[i][j])
        inputs[i][j] = words2id[every_line[i][j]]

