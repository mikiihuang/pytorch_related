import os


corpus_name = "cornell movie-dialogs corpus"

corpus = os.path.join("../data", corpus_name)
print(corpus)

def read_files(file):
    with open(file,"rb") as f:
        contents = f.readlines()
    return contents
filecontents = read_files(corpus+"/movie_lines.txt")
# for line in filecontents:
    # print(type(line))
def get_lines_dict(contents):
    lines = {}
    for line in contents:
        values = str(line).split("+++$+++")
        lines[values[0].replace("b","")] = values[-1].replace("\\n","")
    return lines
id_words  = get_lines_dict(filecontents)
def get_conver_id(contents):
    lines = []
    for line in contents:
        values = str(line).split("+++$+++")
        lines.append(values[-1].replace("\\n",""))
    return lines

conversation = read_files(corpus+"/movie_conversations.txt")
