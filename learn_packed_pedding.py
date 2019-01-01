import torch


with open("test.txt") as f:
    all_lines = []
    words = []
    contents = f.readlines()
    for line in contents:
        line = line.replace(",","").replace(".","").replace("\n","")
        line_token = line.split(" ")
        for word in line_token:
            words.append(word)
        all_lines.append(line_token)
# print(all_lines)
word2id = {word:id for id,word in enumerate(list(set(words)))}
id2word = {id:word for id,word in enumerate(list(set(words)))}
# print(id2word)
