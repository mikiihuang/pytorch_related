import torch
import pooling
import processing
import numpy as np
import matplotlib.pyplot as plt

from classifier.pooling.code import data_loader
import train
model_url  = "yumi"
word2id, id2word = processing.read_Dict("sourceDict.txt")

dev_sentences, dev_lables = processing.reading("../data/cr.dev.txt")
test_sentences,test_lables = processing.reading("../data/cr.test.txt")
dev_x = []
dev_y = []
test_x = []
test_y = []

for i in range(0, len(dev_lables)):
    seq_id = processing.seq2id(dev_sentences[i], word2id)
    target_id = dev_lables[i]
    dev_x.append(seq_id)
    dev_y.append(int(target_id))
for i in range(0, len(test_lables)):
    seq_id = processing.seq2id(test_sentences[i], word2id)
    target_id = dev_lables[i]
    test_x.append(seq_id)
    test_y.append(int(target_id))

dev_x_tensor = data_loader.pad(dev_x, word2id, "-pad-")
dev_y_tensor = torch.from_numpy(np.array(dev_y))

test_x_tensor = data_loader.pad(test_x, word2id, "-pad-")
test_y_tensor = torch.from_numpy(np.array(test_y))
# print(test_x_tensor)
plot_dev = []
plot_test = []
x = []

for i in range(10):
    model = pooling.Pooling(len(word2id), 100, 2)
    model.load_state_dict(torch.load(model_url+"/"+str(i+1)+".pt"))
    correct = 0
    total = 0
    with torch.no_grad():
        outputs1 = model(dev_x_tensor.long())
        dev_right_num = train.accuracy_num(outputs1, dev_y_tensor)
        dev_acc = float(dev_right_num)/len(dev_y_tensor)*100
        outputs2 = model(test_x_tensor.long())
        test_right_num = train.accuracy_num(outputs2, test_y_tensor)
        test_acc = float(test_right_num )/ len(test_y_tensor) * 100
        plot_dev.append(dev_acc)
        plot_test.append(test_acc)
        x.append(i+1)
        print("epoch #"+str(i)+"    dev acc:"+str(dev_acc)+"    test acc:"+str(test_acc))


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






