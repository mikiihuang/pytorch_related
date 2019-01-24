import torch
import torchtext
from torchtext import data,datasets

def tokenizer(text):
    return [i for i in text.split(" ")]

TEXT = data.Field(sequential=True,tokenize=tokenizer,lower=True)
