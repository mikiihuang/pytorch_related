import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

model = nn.Linear(input_size,output_size)
criterior = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

for i in range(num_epochs):
    x_data = torch.from_numpy(x_train)
    y_data = torch.from_numpy(y_train)

    output = model(x_data)
    loss = criterior(output,y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(i+1)%5 == 0:
        print("epoch {}/{},loss:{:.4f}".format(i+1,num_epochs,loss.item()))
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train,y_train,"ro",label = "original data")
plt.plot(x_train,predicted,label = "fitting line")
plt.legend()
plt.show()