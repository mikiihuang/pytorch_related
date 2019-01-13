import torch
import torch.nn as nn
word2idx = {'hello': 0, 'world': 1}
embedding_layer  = nn.Embedding(2,5)
long_tensor = torch.LongTensor([word2idx["hello"]])
# print(long_tensor.type)
embedding_hello = embedding_layer(long_tensor)
print(embedding_hello)
