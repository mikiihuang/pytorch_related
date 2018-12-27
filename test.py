sent_1_x = ['is', 'it', 'too', 'late', 'now', 'say', 'sorry']
sent_1_y = ['VB', 'PRP', 'RB', 'RB', 'RB', 'VB', 'JJ']

sent_2_x = ['ooh', 'ooh']
sent_2_y = ['NNP', 'NNP']

sent_3_x = ['sorry', 'yeah']
sent_3_y = ['JJ', 'NNP']

X = [sent_1_x, sent_2_x, sent_3_x]
Y = [sent_1_y, sent_2_y, sent_3_y]

# map sentences to vocab
vocab = {'<PAD>': 0, 'is': 1, 'it': 2, 'too': 3, 'late': 4, 'now': 5, 'say': 6, 'sorry': 7, 'ooh': 8, 'yeah': 9}
# fancy nested list comprehension
X =  [[vocab[word] for word in sentence] for sentence in X]
# X now looks like:
# [[1, 2, 3, 4, 5, 6, 7], [8, 8], [7, 9]]

tags = {'<PAD>': 0, 'VB': 1, 'PRP': 2, 'RB': 3, 'JJ': 4, 'NNP': 5}
# fancy nested list comprehension
Y =  [[tags[tag] for tag in sentence] for sentence in Y]
# Y now looks like:
# [[1, 2, 3, 3, 3, 1, 4], [5, 5], [4, 5]]

import numpy as np

def padding_input(input,pad_tok):
    every_input_len = [len(sentence) for sentence in input]
    longest_sent = max(every_input_len)
    batch_size = len(input)
    out = np.ones((batch_size,longest_sent))*pad_tok

    for i,x_len in enumerate(every_input_len):
        fill_value = input[i]
        out[i,:x_len] = fill_value[:x_len]
    return out

# X_lengths = [len(sentence) for sentence in X]
# # create an empty matrix with padding tokens
# pad_token = vocab['<PAD>']
# longest_sent = max(X_lengths)
# batch_size = len(X)
# padded_X = np.ones((batch_size, longest_sent)) * pad_token
#
# # copy over the actual sequences
# for i, x_len in enumerate(X_lengths):
#   sequence = X[i]
#   padded_X[i, 0:x_len] = sequence[:x_len]
# print(padded_X)
padding_x = padding_input(X,vocab["<PAD>"])
padding_y = padding_input(Y,tags["<PAD>"])

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

"""
Blog post:
Taming LSTMs: Variable-sized mini-batches and why PyTorch is good for your health:
https://medium.com/@_willfalcon/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
"""


class BieberLSTM(nn.Module):
    def __init__(self, nb_layers, nb_lstm_units=100, embedding_dim=3, batch_size=3):
        self.vocab = {'<PAD>': 0, 'is': 1, 'it': 2, 'too': 3, 'late': 4, 'now': 5, 'say': 6, 'sorry': 7, 'ooh': 8,
                      'yeah': 9}
        self.tags = {'<PAD>': 0, 'VB': 1, 'PRP': 2, 'RB': 3, 'JJ': 4, 'NNP': 5}

        self.nb_layers = nb_layers
        self.nb_lstm_units = nb_lstm_units
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        # don't count the padding tag for the classifier output
        self.nb_tags = len(self.tags) - 1

        # when the model is bidirectional we double the output dimension

        # build actual NN
        self.__build_model()

    def __build_model(self):
        # build embedding layer first
        nb_vocab_words = len(self.vocab)

        # whenever the embedding sees the padding index it'll make the whole vector zeros
        padding_idx = self.vocab['<PAD>']
        self.word_embedding = nn.Embedding(
            num_embeddings=nb_vocab_words,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_idx
        )

        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_lstm_layers,
            batch_first=True,
        )

        # output layer which projects back to tag space
        self.hidden_to_tag = nn.Linear(self.nb_lstm_units, self.nb_tags)

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.hparams.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
        hidden_b = torch.randn(self.hparams.nb_lstm_layers, self.batch_size, self.nb_lstm_units)

        if self.hparams.on_gpu:
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)
