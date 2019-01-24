import torch
import torch.utils.data as Data

import processing
import numpy as np



def pad(data,vocab,pad_character,max_len):
    """
    将不等长的句子进行pad,使之长度相等。
    :param data: list类型
    :param vocab: 自己建的词典，词典中需要包含pad character
    :param pad_character: 使用什么值进行pad
    :return:
    """

    # data_length = [len(item) for item in data]
    # max_len = np.max(data_length)
    # print(max_len)
    pad_num = vocab[pad_character]
    # print(pad_num)

    # 因为我的词典里当pad_character取"-pad-"时，这个pad_num是0,所以可以有下面这一句,要是"-pad-"不是0，不能用下面这句进行pad
    new_array = np.ones((len(data), max_len),dtype=np.int32)*int(pad_num)
    # print(new_array)
    for index, data in enumerate(data):
        # print(index,data)
        # print(data)
        # print(data.dtype)
        new_array[index][:len(data)] = data
    # print(new_array)
    data_tensor = torch.from_numpy(new_array)
    # print(data_tensor.dtype)

    # new_data = []
    # for one in data:
    #     new_data.append(data + [pad_character] * (max_len - len(data)))
    # print(new_data)
    # data_tensor
    return data_tensor



def get_batch(batch,x,y):


    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(dataset=torch_dataset,batch_size=batch,shuffle=True)
    return loader
    # for x_batch, y_batch in loader:
    #     print(x_batch)
        # print("--------")
        # print(y_batch)
    # for epoch in range(10):
    #     for x_batch,y_batch in loader:
    #         batch_x ,batch_y = data
    #         print(i)

    # return batch_x,batch_y

# if __name__ == '__main__':
#     torch.manual_seed(1)
#
#     word2id, id2word = processing.read_Dict("../file/MPQAsourceDict.txt")
#     print(len(word2id))
#     sentences, lables = processing.reading("../data/mpqa.train.txt")
#     train_x = []
#     train_y = []
#
#     for i in range(0, len(lables)):
#         seq_id = processing.seq2id(sentences[i], word2id)
#         target_id = lables[i]
#         train_x.append(seq_id)
#         train_y.append(int(target_id))
#
#     train_x = pad(train_x,word2id,"-pad-")
#     train_x_tensor = pad(train_x)
#     train_y_tensor = torch.from_numpy(np.array(train_y))
#     get_batch(train_x_tensor,train_y_tensor)
#     # train_data = zip(train_x_tensor, train_y_tensor)
#     # print(list(train_data))