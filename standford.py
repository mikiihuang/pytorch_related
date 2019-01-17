import stanfordcorenlp
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'/Users/yumi/Documents/install/standfordNLP/stanford-corenlp-full-2018-10-05',lang="zh")


sentence = '我是重庆工商大学的一名学生'
# print(nlp.word_tokenize(sentence))
# print(nlp.pos_tag(sentence))
print(nlp.ner(sentence))

