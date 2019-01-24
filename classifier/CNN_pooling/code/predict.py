import classifier.CNN_pooling.code.data_loader as data_loader
import CNN
import torch
import processing
import numpy as np
import matplotlib.pyplot as plt
import train
model_url  = "../file/yumi_cnn_mpqa"
word2id, id2word = processing.read_Dict("../file/MPQAsourceDict.txt")

dev_sentences, dev_lables = processing.reading("../../data/mpqa.dev.txt")
test_sentences,test_lables = processing.reading("../../data/mpqa.test.txt")

dev_x,dev_y = processing.get_x_y_list(dev_sentences,dev_lables,word2id)
test_x,test_y = processing.get_x_y_list(test_sentences,test_lables,word2id)

dev_max_len = processing.get_max_length(dev_x)
test_max_len = processing.get_max_length(test_x)
# max_length = max(dev_max_len,test_max_len)

# dev_data_length = [len(item) for item in dev_x]
# max_length = np.max(dev_data_length)
# print(max_length)
#
# test_data_length = [len(item) for item in test_x]
# max_length = np.max(test_data_length)
# print(max_length)


dev_x_tensor = data_loader.pad(dev_x, word2id, "-pad-",38)
dev_y_tensor = torch.from_numpy(np.array(dev_y))

test_x_tensor = data_loader.pad(test_x, word2id, "-pad-",38)
test_y_tensor = torch.from_numpy(np.array(test_y))
# print(test_x_tensor)
plot_dev = []
plot_test = []
x = []
model = CNN.CNNlayer(len(word2id), 100, 2, [3, 4, 5],output_size=2)
with torch.no_grad():
    for i in range(100):
        # model = pooling.Pooling(len(word2id), 100, 2)
        model.eval()
        model.load_state_dict(torch.load(model_url+"/"+str(i+1)+".pt"))
        correct = 0
        total = 0

        outputs1 = model(dev_x_tensor.long())
        dev_right_num = train.accuracy_num(outputs1, dev_y_tensor)
        dev_acc = float(dev_right_num)/len(dev_y_tensor)*100
        outputs2 = model(test_x_tensor.long())
        test_right_num = train.accuracy_num(outputs2, test_y_tensor)
        test_acc = float(test_right_num )/ len(test_y_tensor) * 100
        plot_dev.append(dev_acc)
        plot_test.append(test_acc)
        x.append(i+1)
        print_log = "epoch #{}    dev acc:{}    test acc:{}".format(str(i),str(dev_acc),str(test_acc))
        print(print_log)
        train.write_log("pre_log_38.txt",print_log)

plt.plot(x,plot_dev)
plt.xlabel("epoch")
plt.ylabel("dev acc")
plt.legend()
plt.show()

plt.plot(x,plot_test)
plt.xlabel("epoch")
plt.ylabel("test acc")
plt.legend()
plt.show()




#
#
