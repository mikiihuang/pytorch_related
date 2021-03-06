import pooling
import processing
from classifier.pooling.code import data_loader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
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


    word2id, id2word = processing.read_Dict("../file/SUBJsourceDict.txt")
    sentences, lables = processing.reading("../../data/subj.train.txt")
    train_x = []
    train_y = []
    print_every = 10
    save_every = 40
    for i in range(0, len(lables)):
        seq_id = processing.seq2id(sentences[i], word2id)
        target_id = lables[i]
        train_x.append(seq_id)
        train_y.append(int(target_id))

    train_x_tensor = data_loader.pad(train_x, word2id, "-pad-")
    train_y_tensor = torch.from_numpy(np.array(train_y))

    trainloader = data_loader.get_batch(train_x_tensor, train_y_tensor)

    # model
    poolmodel = pooling.Pooling(len(word2id),100,2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(poolmodel.parameters(),lr=0.01)

    train_loss_plot = []
    train_acc_plot = []
    x_plot = []

    for epoch in range(100):  # loop over the dataset multiple times
        poolmodel.train()
        x_plot.append(epoch+1)
        running_loss = 0.0
        right_num = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            # print(i)
            inputs, targets = data
            batch_size = len(inputs)
            inputs = inputs.long()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = poolmodel(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_right_num = accuracy_num(outputs,targets)
            right_num += train_right_num.tolist()

            if (i+1) % print_every == 0:
                print_log = '[epoch:{},on the {} st batch] loss: {:.8f} accuracy:{:.3f}'.format((epoch + 1), (i + 1), (
                            running_loss / (print_every * batch_size)), (right_num / (print_every * batch_size)))
                print(print_log)
                write_log("subj_log.txt", print_log)

                # print('[epoch:%d,on the %5d st batch] loss: %.8f accuracy:%3f' %(epoch + 1, i + 1, running_loss/(print_every*batch_size),right_num/(print_every*batch_size)))
                if (i+1) % save_every == 0:
                    train_loss_plot.append(running_loss/(print_every*batch_size))
                    train_acc_plot.append(right_num/(print_every*batch_size))
                running_loss = 0
                right_num = 0


        save_model("../file/yumi_subj",poolmodel,str(epoch+1))
    plt.plot(x_plot,train_loss_plot)
    plt.xlabel("epoch")
    plt.ylabel("train_loss")
    plt.show()
    plt.plot(x_plot,train_acc_plot)
    plt.xlabel("epoch")
    plt.ylabel("train_acc")
    plt.show()




