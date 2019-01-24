import torch
import torch.nn as nn
import torch.nn.functional as F

class PoolingLayer(nn.Module):
    def __init__(self,vocab_size,embedding_size,output_size):
        super(PoolingLayer,self).__init__()
        # self,output_size = output_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size,embedding_size)
        self.linearLayer = nn.Linear(embedding_size,output_size)

    def forward(self, x):
        # x的shape为batch_size*sequence_length(一个batch里有几个句子，每个句子由几个词组成)
        embedded = self.embedding(x)
        # 输出的embedded的维度是batch_size * sequence_length * embedding_dim
        # 最大池化就是：对于抽取到若干特征值，只取其中得分最大的那个值作为pooling层保留值
        # embedding后得到的就是句子特征值了
        # 取最大池化感觉是对每一个句子取最大池化，核函数的取值应该是sequence_length，相当于在多大的窗口里取池化
        # 按照官方文档的说明，输入的维度是(N,C,Lin)，输出是(N,C,Lout)。
        # 所以需要对embedded的第二维和第三维进行转换。
        embedded = torch.transpose(embedded,1,2)
        # 现在embedded的维度是batch_size * embedding_dim *sequence_length
        # embedded = embedded.view(1,embedded.size(2),embedded.size(1))
        out = F.max_pool1d(embedded,kernel_size = embedded.size(2))
        out = out.squeeze(2)
        out = self.linearLayer(out)
        return out
#
class Pooling(nn.Module):
    def __init__(self, vocab_size, embedding_size,label_vocab):
        super(Pooling, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embedding_size
        self.label_num = label_vocab
        self.embed_dropout = 0.5
        self.fc_dropout = 0.5

        self.embeddings = nn.Embedding(self.vocab_size, self.embed_dim)

        self.linear1 = nn.Linear(self.embed_dim, self.embed_dim // 2)
        self.linear2 = nn.Linear(self.embed_dim // 2, self.label_num)
        self.embed_dropout = nn.Dropout(self.embed_dropout)
        self.fc_dropout = nn.Dropout(self.fc_dropout)


    def forward(self, input):
        out = self.embeddings(input)
        out = torch.tanh(out)
        out = self.embed_dropout(out)
        out = torch.transpose(out, 1, 2)
        out = F.max_pool1d(out, out.size(2))
        out = out.squeeze(2)
        out = self.fc_dropout(out)
        out = self.linear1(F.relu(out))
        out = self.linear2(F.relu(out))
        return out

pool = Pooling(vocab_size=200,embedding_size=100,label_vocab=2)
print(pool)