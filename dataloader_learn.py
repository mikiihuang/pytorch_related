import torch
import torch.utils.data as Data
torch.manual_seed(1)

BATCH_SIZE = 5
x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)

torch_dataset = Data.TensorDataset(x,y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
# 将所有的数据训练了３次
for epoch in range(3):
    #总共有10条数据,batch_size是５,so training for 2 times
    for step,(batch_x,batch_y) in enumerate(loader):
        print("Epoch:",epoch,"|step:",step,"|batch x:",batch_x.numpy(),"|batch y:",batch_y.numpy())
print("\n")
# if batch_size = 8,the second training size in each epoch will be the remaining data
# eg.
torch_dataset2 = Data.TensorDataset(x,y)
loader2 = Data.DataLoader(
    dataset=torch_dataset2,
    batch_size= 8 ,
    shuffle=True
)
for epoch in range(3):
    # 总共有10条数据,batch_size是５,so training for 2 times
    for step, (batch_x, batch_y) in enumerate(loader2):
        print("Epoch:", epoch, "|step:", step, "|batch x:", batch_x.numpy(), "|batch y:", batch_y.numpy())