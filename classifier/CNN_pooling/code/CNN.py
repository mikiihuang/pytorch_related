
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNlayer(nn.Module):
    def __init__(self,vocab_size,embedding_size,kernel_num,kernel_Size,output_size):
        super(CNNlayer,self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.kernel_num = kernel_num
        self.kernel_size = kernel_Size

        self.output_size = output_size
        self.embedded = nn.Embedding(self.vocab_size,self.embedding_size)



        # self.convs = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv1d(in_channels=embedding_size, out_channels=kernel_num, kernel_size=one),
        #         nn.ReLU(),
        #         # 因为默认步长是1且没有padding，所以池化时的kernel_size就是Lout的维度。
        #         nn.MaxPool1d(kernel_size=max_seq_len-one+1)
        #     )
        #     for one in kernel_Size
        # ])
        self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels=self.embedding_size, out_channels=self.kernel_num, kernel_size=self.kernel_size[0]),
                nn.ReLU(),
                # 因为默认步长是1且没有padding，所以池化时的kernel_size就是Lout的维度。
                # nn.MaxPool1d(kernel_size=int(max_seq_len-self.kernel_size[0]+1))
            )

        self.conv2 = nn.Sequential(
                nn.Conv1d(in_channels=self.embedding_size, out_channels=self.kernel_num,
                          kernel_size=self.kernel_size[1]),
                nn.ReLU(),
                # 因为默认步长是1且没有padding，所以池化时的kernel_size就是Lout的维度。
                # nn.MaxPool1d(kernel_size=int(max_seq_len - self.kernel_size[1] + 1))
            )
        self.conv3 = nn.Sequential(
                nn.Conv1d(in_channels=self.embedding_size, out_channels=self.kernel_num,
                          kernel_size=self.kernel_size[2]),
                nn.ReLU(),
                # 因为默认步长是1且没有padding，所以池化时的kernel_size就是Lout的维度。
                # nn.MaxPool1d(kernel_size=int(max_seq_len - self.kernel_size[2] + 1))
            )
        self.embedding_dropout = nn.Dropout()
        self.fcdropout = nn.Dropout()

        # in_features的维度，看那张图可以知道是拼接几个特征之后的，那几个特征由先卷积再池化得到，所以是feature map的数量,不同size的filter的个数这两者的乘积。
        in_feature = self.kernel_num*len(self.kernel_size)
        self.linear1 = nn.Linear(in_features=in_feature,out_features=in_feature//2)
        self.linear2 = nn.Linear(in_features=in_feature//2,out_features=output_size)
    def forward(self,x):

        out = self.embedded(x)
        out = self.embedding_dropout(out)
        out = torch.transpose(out,1,2)
        out1 = self.conv1(out)
        out2 = self.conv2(out)
        out3 = self.conv3(out)
        out1 = F.max_pool1d(out1, kernel_size=out1.size(2))
        out2 = F.max_pool1d(out2, kernel_size=out2.size(2))
        out3 = F.max_pool1d(out3, kernel_size=out3.size(2))


        out = torch.cat((out1, out2, out3), 1).squeeze(2)
        # print(out.shape)
        #
        # out = [conv(out).squeeze(2) for conv in self.convs]
        # # 上一步的out中每一个tensor的维度都是(5,2,1),经过cat之后维度变成(5,6,1)
        # out = torch.cat(out,dim=1)
        out = self.fcdropout(out)
        out = self.linear1(F.relu(out))
        out = self.linear2(F.relu(out))
        return out

#
# model = CNNlayer(3000,3,2,[2,3,4],2)
# # model = CNNlayer(vocab_size=8000,embedding_size=100,kernel_num=2,kernel_Size=[2,3,4],max_seq_len=45,output_size=2)
# # inputs = torch.rand(5,4)
# inputs = np.array(
#     [[1,2,3,4],
#      [2,2,3,4],
#      [3,2,3,4],
#      [4,2,3,4],
#      [5,2,3,4]
#     ]
# )
# inputs = torch.from_numpy(inputs)
# # print(inputs.dtype)
#
# # print(test)
# # print(model)
# pred = model(inputs)
# print(pred)



