from collections import defaultdict
import time
import random
import torch
from tqdm import tqdm

class CNNclass(torch.nn.Module):
    def __init__(self, nwords, emb_size, num_filters, window_size, ntags):
        super(CNNclass, self).__init__()

        """ layers """
        self.embedding = torch.nn.Embedding(nwords, emb_size)
        # uniform initialization
        torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        # Conv 1d
        self.conv_1d = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=window_size,
                                       stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.relu = torch.nn.ReLU()
        self.projection_layer = torch.nn.Linear(in_features=num_filters, out_features=ntags, bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)

    def forward(self, words):
        emb = self.embedding(words)                 # batch_size x nwords x emb_size
        emb = emb.permute(0, 2, 1)                  # batch_size x emb_size x nwords
        h = self.conv_1d(emb)                       # batch_size x num_filters x nwords
        # Do max pooling
        h = h.max(dim=2)[0]                         # batch_size x num_filters
        h = self.relu(h)
        out = self.projection_layer(h)              # size(out) = batch_size x ntags
        return out

type = torch.LongTensor
use_cuda = torch.cuda.is_available()
print('cuda:', use_cuda)
if use_cuda:
    type = torch.cuda.LongTensor

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]

# Define the model
EMB_SIZE = 64
WIN_SIZE = 3
FILTER_SIZE = 64

def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            words = [w2i[x] for x in words.split(" ")] 
            if len(words) < WIN_SIZE:
                words += [0] * (WIN_SIZE - len(words))
            yield (torch.tensor(words).type(type), t2i[tag])


# Read in the data
train = list(read_dataset("topicclass/topicclass_train.txt"))
valid = list(read_dataset("topicclass/topicclass_valid.txt"))
test = list(read_dataset("topicclass/topicclass_test.txt"))
train_size, valid_size, test_size = len(train), len(valid), len(test)
w2i = defaultdict(lambda: UNK, w2i)
nwords, ntags = len(w2i), len(t2i)
print('Data loaded')
# initialize the model
model = CNNclass(nwords, EMB_SIZE, FILTER_SIZE, WIN_SIZE, ntags)
if use_cuda:
    model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

batch_size = 256
for ITER in range(100):
    # Perform training
    random.shuffle(train)
    train_loss = 0.0
    train_correct = 0.0
    start = time.time()
    for index in tqdm(range(0, train_size, batch_size)):
        words, tag = zip(*train[index:index+batch_size])
        words_tensor = torch.nn.utils.rnn.pad_sequence(words, batch_first=True)
        tag_tensor = torch.tensor(tag).type(type)
        scores = model(words_tensor)
        predict = scores.argmax(dim=1)
        train_correct += (predict == tag_tensor).sum().item()
        my_loss = criterion(scores, tag_tensor)
        train_loss += my_loss.sum().item()
        # Do back-prop
        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()
    print("iter %r: train loss/sent=%.4f, acc=%.4f, time=%.2fs" % (
        ITER+1, train_loss / train_size, train_correct / train_size, time.time() - start))
    # Perform testing
    valid_correct = 0.0
    for index in range(0, valid_size, batch_size):
        words, tag = zip(*valid[index:index+batch_size])
        words_tensor = torch.nn.utils.rnn.pad_sequence(words, batch_first=True)
        tag_tensor = torch.tensor(tag).type(type)
        scores = model(words_tensor)
        predict = scores.argmax(dim=1)
        valid_correct += (predict == tag_tensor).sum().item()
    print("iter %r: valid acc=%.4f" % (ITER+1, valid_correct / valid_size))
