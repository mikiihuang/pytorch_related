import torchtext
import pandas as pd
import numpy as np
from torchtext import data,datasets

import re
train_path = "../data/train.csv"
val_path =  "../data/valid.csv"
test_path = "../data/test.csv"

# df_train = pd.read_csv("../data/train.csv")
# df_test = pd.read_csv("../data/test.csv")
# df_val = pd.read_csv("../data/valid.csv")

#
def tokenizer(comment):
    # comment = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(comment))
    # comment = re.sub(r"[]+", " ",comment)
    # comment = re.sub(r"\!+", "!", comment)
    # comment = re.sub(r"\,+", ",", comment)
    # comment = re.sub(r"\?+", "?", comment)

    return [i for i in comment.split(" ")]
tokenizer = lambda x: x.split()

TEXT = data.Field(sequential = True,tokenize=tokenizer,lower=True)
LABLE = data.Field(sequential = False,use_vocab=False)

train_val_fields = [("id",None),
                ("comment_text",TEXT),
                ("toxic",LABLE),
                ("severe_toxic",LABLE),
                ("obscene",LABLE),
                ("threat",LABLE),
                ("insult",LABLE),
                ("identity_hate",LABLE)]
test_fileds = [("id",None),
               ("comment_text",TEXT)]

train_data,val_data= data.TabularDataset.splits(path = "../data",format="csv",train = "train.csv",validation = "valid.csv",fields=train_val_fields,skip_header=True)
test_data = data.TabularDataset(path=test_path,format="csv",fields=test_fileds,skip_header=True)

# print(TEXT.build_vocab(train_data))
# for i in range(len(train_data)):
#     print(vars(train_data[i]))


# 查看Dataset内部
print(train_data[0].comment_text)
print(train_data[0].__dict__.keys())


TEXT.build_vocab(train_data)
vocab = TEXT.vocab

print(vars(vocab))



