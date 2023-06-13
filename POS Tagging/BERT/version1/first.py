import numpy as np
import pickle

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from pytorch_pretrained_bert import BertTokenizer, BertModel

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys

import time

global start
start = time.time()

sys.path.append('../../../Parsing/')
print(os.getcwd())
from language_elements import Sentence, Word, Chunk

with open('../../../Parsing/full_dataset_113.pickle', 'rb') as file:
    retrieved_sentences = pickle.load(file)

tagged_sentences = []

for sentence in retrieved_sentences:
    temp = []
    for chunk in sentence.list_of_chunks:
        for word in chunk.list_of_words:
            temp.append((word.kannada_word, word.pos))
    tagged_sentences.append(temp)

tags = list(set(word_pos[1] for sentence in tagged_sentences for word_pos in sentence))
tags = ["<pad>"] + tags

tag2index = {tag:idx for idx, tag in enumerate(tags)}
index2tag = {idx:tag for tag, idx in enumerate(tags)}

train_data, test_data = train_test_split(tagged_sentences, test_size=0.1)
print("No. of sentences in train data:", len(train_data), "\nNo. of sentences in test data:", len(test_data))

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda'

# How to use tokenizer for this
    # especially for Kannada
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

class PosDataset(data.Dataset):
    def __init__(self, tagged_sentences):
        self.list_of_sentences  = []
        self.list_of_tags  = []

        for sentence in tagged_sentences:
            words, tags = zip(*sentence)
            words, tags = list(words), list(tags)

            self.list_of_sentences.append(["[CLS]"] + words + ["[SEP]"])
            self.list_of_tags.append(["<pad>"] + tags + ["<pad>"])

    def __len__(self):
        return len(self.list_of_sentences)
    
    def __getitem__(self, index):
        words, tags = self.list_of_sentences[index], self.list_of_tags[index]

        x, y = [], []
        is_heads = []

        for w, t in zip(words, tags):
            # tokens = tokenizer.tokenize()
            tokens = tokenizer.tokenize(w) if w not in ('[CLS]', '[SEP]') else [w]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0] * (len(tokens) - 1)

            t = [t] + ["<pad>"] * (len(tokens) - 1)
            y_ids = [tag2index[i] for i in t]

            x.extend(token_ids)
            is_heads.extend(is_head)
            y.extend(y_ids)

        assert len(x)==len(y)==len(is_heads)
        # print("Length of x: ", len(x))
        # print("Length of y: ", len(y))
        # print("Length of is_heads: ", len(is_heads))

        seqlen = len(y)

        words = " ".join(words)
        tags = " ".join(tags)

        return words, tags, is_heads, tags, y, seqlen
    
def pad(batch):
    # print("=================================")
    # for i in batch:
    #     print(i)
    #     print("=================================")
    # print(len(batch))

    # print("=================================")
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(2, maxlen)
    y = f(-2, maxlen)


    f = torch.LongTensor

    return words, f(x), is_heads, tags, f(y), seqlens

# Model

class Net(nn.Module):
    def __init__(self, vocab_size=None):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')

        self.fc = nn.Linear(768, vocab_size)
        self.device = device

    def forward(self, x, y):
        '''
        x: (N, T). int64
        y: (N, T). int64
        '''
        x = x.to(device)
        y = y.to(device)
        
        if self.training:
            self.bert.train()
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
                enc = encoded_layers[-1]
        
        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat

def train(model, iterator, optimizer, criterion):
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        _y = y # for monitoring
        optimizer.zero_grad()
        logits, y, _ = model(x, y) # logits: (N, T, VOCAB), y: (N, T)

        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        if i%10==0: # monitoring
            global start
            print("step: {}, loss: {}, time: {}".format(i, loss.item(), time.time()-start))
            start = time.time()
    
def eval(model, iterator):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch

            _, _, y_hat = model(x, y)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    ## gets results and save
    with open("result", 'w') as fout:
        count = 0
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            try:
                y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
                preds = [index2tag[hat] for hat in y_hat]
                assert len(preds)==len(words.split())==len(tags.split())
                for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                    fout.write("{} {} {}\n".format(w, t, p))
                fout.write("\n")
            except:
                count+=1
        print("Count of errors:", count)

            
    ## calc metric
    y_true =  np.array([index2tag[line.split()[1]] for line in open('result', 'r').read().splitlines() if len(line) > 0])
    y_pred =  np.array([index2tag[line.split()[2]] for line in open('result', 'r').read().splitlines() if len(line) > 0])

    acc = (y_true==y_pred).astype(np.int32).sum() / len(y_true)

    print("acc=%.2f"%acc)

model = Net(vocab_size=len(tag2index))
model.to(device)
model = nn.DataParallel(model)

train_dataset = PosDataset(train_data)
eval_dataset = PosDataset(test_data)

train_iter = data.DataLoader(dataset=train_dataset,
                             batch_size=8,
                             shuffle=True,
                             collate_fn=pad)

test_iter = data.DataLoader(dataset=eval_dataset,
                             batch_size=8,
                             shuffle=False,
                             collate_fn=pad)

optimizer = optim.Adam(model.parameters(), lr = 0.001)
# increased learning rate by 10 times
criterion = nn.CrossEntropyLoss(ignore_index=0)

train(model, train_iter, optimizer, criterion)
eval(model, test_iter)


open('result', 'r').read().splitlines()[:100]