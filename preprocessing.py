from collections import defaultdict
import pickle
import numpy as np

MIN_FREQ = 3
EMBED_DIM = 300

w2f = defaultdict(lambda: 0)
def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            for x in words.split(" "):
                w2f[x] += 1

read_dataset("topicclass/topicclass_train.txt")
read_dataset("topicclass/topicclass_valid.txt")
read_dataset("topicclass/topicclass_test.txt")

w2i = {'<unk>': 0}
for k, v in w2f.items():
    if v >= MIN_FREQ and k != '<unk>':
        w2i[k] = len(w2i)
pickle.dump(w2i, open('w2i.pkl', 'wb'))
VOCAB_SIZE = len(w2i)
print('%d word saved.' % VOCAB_SIZE)

embedding = (np.random.rand(VOCAB_SIZE, EMBED_DIM) - 0.5) / 2

from tqdm import tqdm
embed_load = 0
with open('crawl-300d-2M.vec') as f:
    N, D = f.readline()[:-1].split(' ')
    N, D = int(N), int(D)
    for i in tqdm(range(N)):
        line = f.readline()[:-1]
        token = line[:line.find(' ')]
        if token in w2i:
            embedding[w2i[token]] = np.array([float(i) for i in line[line.find(' ')+1:].split(' ')])
            embed_load += 1
np.save('embedding', embedding)
print('%d pretrained loaded.' % embed_load)

