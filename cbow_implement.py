import torch
import torch.nn as nn
import torch.nn.functional as F

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


class CBOW(nn.Module):

    def __init__(self,vocab_size,embbeding_size,hidden_size):
        super(CBOW,self).__init__()
        self.embedded = nn.Embedding(vocab_size,embbeding_size)
        self.linear1 = nn.Linear(4*embbeding_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,vocab_size)

    def forward(self, inputs):
        embedding = self.embedded(inputs).view(1,-1)
        out = self.linear1(embedding)
        out = F.relu(out)
        out = self.linear2(out)

# create your model and train.  here are some functions to help you make
# the data ready for use by your module


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


make_context_vector(data[0][0], word_to_ix)  # exampl