import re
import collections
import numpy as np

def reading(filename):
    '''
    读取文件内容，去掉标点符号
    :param filename:
    :return: [["i",'will','go'...],
            ["this","is",...]]
    '''
    with open(filename,encoding="utf-8") as f:
        contents = f.readlines()
    labels = []
    sentences = []
    for line in contents:
        # print(line.strip().split(" "))
        # print(line.strip())
        line = line.strip()
        after_split = line.split(" ")
        labels.append(after_split[0])
        punc = [",", ".", "'", "[", "]", "(", ")", "/", "\\", ":", ""]
        # re.findall("\d+",i)
        every_sentence = []
        for i in after_split[2:]:
            if i not in punc:
                every_sentence.append(i)
        sentences.append(every_sentence)
    # print(sentences)
    return sentences,labels
    # print(labels)


def refreshData(string):
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)



def create_vocab(words_list):
    words2id = collections.OrderedDict()
    words2id["-pad-"] = len(words2id)
    words2id["-unk-"] = len(words2id)
    for sentence in words_list:
        for word in sentence:
            if word not in words2id:
                words2id[word] = len(words2id)

    # print(words2id)
    # print(type(words2id))
    return words2id

# print(vocab["i"])
# print(type(vocab))
# print(dict(vocab))
def vocab_to_file(vocab,filename):
    with open(filename,"w",encoding="utf-8") as f:
        for key in vocab:
            # f.write()
            f.write(key+ " "+str(vocab[key]))
            f.write("\n")

def read_Dict(filename):
    with open(filename,encoding="utf-8") as f:
        contents = f.readlines()
    word2id = collections.OrderedDict()
    id2word = collections.OrderedDict()
    for line in contents:
        line = line.strip().split(" ")
        word2id[line[0]] = int(line[1])
        id2word[line[1]] = line[0]
    return word2id,id2word


def seq2id(sentence,vocab):
    sequence_id = []
    punc = [",", ".", "'", "[", "]", "(", ")", "/", "\\", ":", ""]
    for word in sentence:
        if word not in punc:
            if word in vocab:
                sequence_id.append(vocab[word])
            else:
                sequence_id.append(vocab["-unk-"])
    return sequence_id

def get_x_y_list(sentence,lable,word2id):
    x = []
    y = []
    for i in range(0, len(lable)):
        seq_id = seq2id(sentence[i], word2id)
        target_id = lable[i]
        x.append(seq_id)
        y.append(int(target_id))

    return x,y

def get_max_length(x):
    data_length = [len(item) for item in x]
    max_length = np.max(data_length)
    return max_length

if __name__ == '__main__':
    train_sentences, train_lables = reading("../../data/mpqa.train.txt")
    # dev_sentences,dev_labels = reading("../../data/subj.dev.txt")
    # test_sentences,test_labels = reading("../../data/subj.test.txt")
    #
    # sentences = train_sentences+dev_sentences+test_sentences
    # print(sentences)
    # print(len(sentences))
    sentences = train_sentences

    # # print(len(lables))
    vocab = create_vocab(sentences)
    vocab_to_file(vocab,"../file/MPQAsourceDict.txt")
    word2id,id2word = read_Dict("../file/MPQAsourceDict.txt")
    # print(word2id)
    # print(id2word)
    # for i in range(0,len(lables)):
    #     seq_id = seq2id(sentences[i], word2id)
    #     target_id = lables[i]
    #     print(seq_id)
    #     print(target_id)

