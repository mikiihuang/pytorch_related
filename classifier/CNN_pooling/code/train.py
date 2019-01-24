import CNN
import processing
import data_loader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import CNN
import os


def write_log(filename,things):
    with open(filename,"a") as f:
        f.write(things+"\n")


def accuracy_num(pred,lable):
    '''

    :param pred:tensor类型，预测值
    :param lable: tensor类型，标注值
    :return: tensor int64类型
    '''
    pred = torch.tensor(pred)
    lable = torch.tensor(lable)
    pred_label = torch.max(pred,1)[1]
    count_num = (pred_label==lable).sum()
    return count_num

def save_model(folder,network_name,model_name):
    '''

    :param url: 保存路径
    :param name: 保存模型的名字
    :return: 无
    '''
    if not os.path.isdir(folder):
        os.makedirs(folder)
    name = folder+"/"+model_name+".pt"
    torch.save(network_name.state_dict(),name)






if __name__ == '__main__':


    word2id, id2word = processing.read_Dict("../file/MPQAsourceDict.txt")
    train_sentences, train_lables = processing.reading("../../data/mpqa.train.txt")
    dev_sentences, dev_lables = processing.reading("../../data/mpqa.dev.txt")
    test_sentences, test_lables = processing.reading("../../data/mpqa.test.txt")

    train_x,train_y = processing.get_x_y_list(train_sentences,train_lables,word2id)
    dev_x,dev_y = processing.get_x_y_list(dev_sentences,dev_lables,word2id)
    test_x, test_y = processing.get_x_y_list(test_sentences, test_lables, word2id)
    print_every = 30
    save_every = 120


    # 获取训练集、验证集、测试集中最大句子的长度
    max_length = processing.get_max_length(train_x)
    # dev_length = processing.get_max_length(dev_x)
    # test_length = processing.get_max_length(test_x)
    #
    # max_length = max(train_length,dev_length,test_length)
    # # print(max_length)




    train_x_tensor = data_loader.pad(train_x, word2id, "-pad-",max_length)
    train_y_tensor = torch.from_numpy(np.array(train_y))

    trainloader = data_loader.get_batch(64,train_x_tensor, train_y_tensor)

    # model
    cnnModel = CNN.CNNlayer(len(word2id),100,2,[3,4,5],output_size=2)
    # poolmodel = CNN.CNN(len(word2id),100,2,[3,4],1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnnModel.parameters(),lr=0.01)

    train_loss_plot = []
    train_acc_plot = []
    x_plot = []

    for epoch in range(100):  # loop over the dataset multiple times
        cnnModel.train()
        x_plot.append(epoch+1)
        running_loss = 0.0
        right_num = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            # print(i)
            inputs, targets = data
            # print(inputs)
            # print(type(inputs))
            batch_size = len(inputs)
            inputs = inputs.long()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnnModel(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_right_num = accuracy_num(outputs,targets)
            right_num += train_right_num.tolist()

            if (i+1) % print_every == 0:
                print_log = '[epoch:{},on the {} st batch] loss: {:.8f} accuracy:{:.3f}'.format((epoch+1),(i+1),(running_loss/(print_every*batch_size)),(right_num/(print_every*batch_size)))
                print(print_log)
                write_log("log.txt",print_log)
                # print('[epoch:%d,on the %5d st batch] loss: %.8f accuracy:%3f' %(epoch + 1, i + 1, running_loss/(print_every*batch_size),right_num/(print_every*batch_size)))
                if (i+1) % save_every == 0:
                    train_loss_plot.append(running_loss/(print_every*batch_size))
                    train_acc_plot.append(right_num/(print_every*batch_size))
                running_loss = 0

                right_num = 0


        save_model("../file/yumi_cnn_mpqa",cnnModel,str(epoch+1))
    plt.plot(x_plot,train_loss_plot)
    plt.xlabel("epoch")
    plt.ylabel("train_loss")
    plt.show()
    plt.plot(x_plot,train_acc_plot)
    plt.xlabel("epoch")
    plt.ylabel("train_acc")
    plt.show()




