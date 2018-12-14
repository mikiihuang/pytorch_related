import jieba
import gensim
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import re
import jieba.posseg as pos
import matplotlib.pyplot as plt

texts = '明天是荣耀运营十周年纪念日。' \
     '荣耀从两周年纪念日开始，' \
     '在每年的纪念日这天凌晨零点会开放一个新区。' \
     '第十版账号卡的销售从三个月前就已经开始。' \
     '在老区玩的不顺心的老玩家、准备进入荣耀的新手，都已经准备好了新区账号对这个日子翘首以盼。' \
     '陈果坐到了叶修旁边的机器，随手登录了她的逐烟霞。' \
     '其他九大区的玩家人气并没有因为第十区的新开而降低多少，' \
     '越老的区越是如此，实在是因为荣耀的一个账号想经营起来并不容易。' \
     '陈果的逐烟霞用了五年时间才在普通玩家中算是翘楚，哪舍得轻易抛弃。' \
     '更何况到最后大家都会冲着十大区的共同地图神之领域去。'
texts = texts.replace("。","").replace("，","").replace("、","")

words_tags = pos.cut(texts)
words = []
tags = []
for word,tag in words_tags:
    words.append(word)
    tags.append(tag)
# print(words)
words_tags_list = [(words,tags)]
# print(words_tags_zip)
# for i in words_tags_zip:
#     print(i)
word_dict = gensim.corpora.Dictionary([words])
word2id=word_dict.token2id
# print(word_dict)
# print(word2id)
tag_dict = gensim.corpora.Dictionary([tags])
tag2id = tag_dict.token2id
# print(word_dict)

# 构造Lstm类
class LSTMPos(nn.Module):
    def __init__(self,vocab_size,embbeding_size,hidden_size,output_size):
        super(LSTMPos,self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size,embbeding_size)
        self.lstm = nn.LSTM(embbeding_size,hidden_size)
        self.linear = nn.Linear(hidden_size,output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1,1,self.hidden_size),
                torch.zeros(1,1,self.hidden_size))

    def forward(self, inputs):
        embedding = self.embed(inputs.view(1,-1))
        out,_ = self.lstm(embedding.view(len(inputs),1,-1),self.hidden)
        out = self.linear(out.view(len(inputs),-1))
        tags = F.log_softmax(out,dim=1)
        return tags
model = LSTMPos(len(word2id),10,10,len(tag2id))
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

# Embedding的实例接收的是LongTensor
# inputs = Variable(torch.LongTensor(list(map(lambda x:word2id[x],words))))

total_loss = []
for i in range(300):
    loss_1epoch = 0
    for word,tag in words_tags_list:
        optimizer.zero_grad()
        model.hidden = model.init_hidden()

        sentence_in = Variable(torch.LongTensor(list(map(lambda x:word2id[x],word))))
        targets = Variable(torch.LongTensor(list(map(lambda x:tag2id[x],tag))))

        tag_pred = model(sentence_in)
        loss = criterion(tag_pred,targets)
        loss.backward()
        optimizer.step()
        loss_1epoch += loss.item()
    total_loss.append(loss_1epoch)
    print("step {}  loss:{}".format((i+1),loss_1epoch))

plt.plot(total_loss,"r-")
plt.show()
