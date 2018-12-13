import torch
from torch.autograd import Variable
import torch.nn.functional as F

# data = "this is the first try in pytorch"
# data = data.split()
# print(data)
# word2id = {word:id for id,word in enumerate(data)}
# print(word2id)
#
# embedding = torch.nn.Embedding(len(word2id),3)
# tensor = torch.tensor([word2id["this"]],dtype = torch.long)
#
# print(embedding(tensor))

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

no_repeat = set(test_sentence)
word2index = {word:id for id,word in enumerate(list(no_repeat))}
id2word = {id:word for id,word in enumerate(list(no_repeat))}
print(id2word)
# print(word2index)
print(len(no_repeat))
trigrams = [([test_sentence[i],test_sentence[i+1]],test_sentence[i+2]) for i in range(len(test_sentence)-2)]
# print(trigrams[:3])
# print(len(trigrams))



class Ngram(torch.nn.Module):
    # n_gram是上下文长度,表示我们想由前面的几个单词来预测这个单词
    def __init__(self,vocab_size,embedding_size,n_gram,hidden_size):
        super(Ngram,self).__init__()
        self.embedded = torch.nn.Embedding(vocab_size,embedding_size)
        self.linear1 = torch.nn.Linear(n_gram*embedding_size,hidden_size)

        # 为什么输出的维度是vocab_size??????
        # 输出的维数是单词总数，可以看成一个分类问题，要最大化预测单词的概率
        self.linear2 = torch.nn.Linear(hidden_size,vocab_size)
    def forward(self, x):
        embed = self.embedded(x).view((1,-1))
        out = F.relu(self.linear1(embed))
        out = self.linear2(out)
        # 最后经过一个log softmax激活函数
        log_probs = F.log_softmax(out,dim=1)
        return log_probs
losses = []
criterion = torch.nn.NLLLoss()
model = Ngram(len(no_repeat),EMBEDDING_DIM,2,128)
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

# 跑了100遍我全部的数据
for epoch in range(100):
    total_loss = 0
    # 对全部数据中的每一条数据
    for context,target in trigrams:
        context_ids = Variable(torch.LongTensor(list(map(lambda w:word2index[w],context))))
        # context_ids = torch.tensor([word2index[w] for w in context],dtype = torch.long)
        out = model(context_ids)
        loss = criterion(out,torch.tensor([word2index[target]],dtype = torch.long))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    losses.append(total_loss)
print(losses)
word, label = trigrams[3]
print(word,label)
word = Variable(torch.LongTensor([word2index[i] for i in word]))
out = model(word)
# 当max函数中有维数参数的时候，它的返回值为两个，一个为最大值，另一个为最大值的索引
_, predict_label = torch.max(out, 1)
print(predict_label)
print(predict_label.long())
predict_word = id2word[int(predict_label.numpy())]
print(predict_word)
# print(\'real word is {}, predict word is {}\'.format(label, predict_word))