#!/usr/bin/env python
# coding: utf-8

# [BERT](https://arxiv.org/abs/1810.04805) is known to be good at Sequence tagging tasks like Named Entity Recognition. Let's see if it's true for POS-tagging.

# In[33]:


__author__ = "kyubyong"
__address__ = "https://github.com/kyubyong/nlp_made_easy"
__email__ = "kbpark.linguist@gmail.com"


# In[34]:


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from tqdm import tqdm_notebook as tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer


# In[35]:


torch.__version__


# # Data preparation

# Thanks to the great NLTK, we don't have to worry about datasets. Some of Penn Tree Banks are included in it. I believe they serves for the purpose.

# In[36]:


import nltk
nltk.download('treebank')
tagged_sents = nltk.corpus.treebank.tagged_sents()
len(tagged_sents)


# In[37]:


tagged_sents[0]


# In[38]:


tags = list(set(word_pos[1] for sent in tagged_sents for word_pos in sent))


# In[39]:


",".join(tags)


# In[40]:


# By convention, the 0'th slot is reserved for padding.
tags = ["<pad>"] + tags


# In[41]:


tag2idx = {tag:idx for idx, tag in enumerate(tags)}
idx2tag = {idx:tag for idx, tag in enumerate(tags)}


# In[42]:


# Let's split the data into train and test (or eval)
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(tagged_sents, test_size=.1)
len(train_data), len(test_data)


# In[43]:

print("===================================================================================")
print(torch.cuda.is_available())
print("===================================================================================")


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# # Data loader
# 

# In[44]:


tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)


# In[45]:


class PosDataset(data.Dataset):
    def __init__(self, tagged_sents):
        sents, tags_li = [], [] # list of lists
        for sent in tagged_sents:
            words = [word_pos[0] for word_pos in sent]
            tags = [word_pos[1] for word_pos in sent]
            sents.append(["[CLS]"] + words + ["[SEP]"])
            tags_li.append(["<pad>"] + tags + ["<pad>"])
        self.sents, self.tags_li = sents, tags_li

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx] # words, tags: string list

        # We give credits only to the first piece.
        x, y = [], [] # list of ids
        is_heads = [] # list. 1: the token is the first piece of a word
        for w, t in zip(words, tags):
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0]*(len(tokens) - 1)

            t = [t] + ["<pad>"] * (len(tokens) - 1)  # <PAD>: no decision
            yy = [tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)

        assert len(x)==len(y)==len(is_heads), "len(x)={}, len(y)={}, len(is_heads)={}".format(len(x), len(y), len(is_heads))

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen


# In[46]:


def pad(batch):
    # for b in batch:
    #     print("-------------------------------------------------------------")
    #     print(b)
    #     print("-------------------------------------------------------------")
    
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    
    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)


    f = torch.LongTensor

    return words, f(x), is_heads, tags, f(y), seqlens


# # Model

# In[47]:


from pytorch_pretrained_bert import BertModel


# In[48]:


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


# # Train an evaluate

# In[49]:


def train(model, iterator, optimizer, criterion):
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        _y = y # for monitoring

        # Move the model to cuda:0 if not already on that device
        model = model.to('cuda:0')

        # Move the input tensor to cuda:0 if not already on that device
        x = x.to('cuda:0')
        optimizer.zero_grad()
        logits, y, _ = model(x, y) # logits: (N, T, VOCAB), y: (N, T)

        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        if i%100==0:
            print("step: {}, loss: {}".format(i, loss.item()))


# In[50]:


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
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds)==len(words.split())==len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write("{} {} {}\n".format(w, t, p))
            fout.write("\n")
            
    ## calc metric
    y_true =  np.array([tag2idx[line.split()[1]] for line in open('result', 'r').read().splitlines() if len(line) > 0])
    y_pred =  np.array([tag2idx[line.split()[2]] for line in open('result', 'r').read().splitlines() if len(line) > 0])

    acc = (y_true==y_pred).astype(np.int32).sum() / len(y_true)

    print("acc=%.2f"%acc)


# ## Load model and train

# In[51]:


model = Net(vocab_size=len(tag2idx))
model.to(device)
model = nn.DataParallel(model)


# In[52]:


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

optimizer = optim.Adam(model.parameters(), lr = 0.0001)

criterion = nn.CrossEntropyLoss(ignore_index=0)


# In[ ]:


train(model, train_iter, optimizer, criterion)
eval(model, test_iter)


# Check the result.

# In[ ]:


open('result', 'r').read().splitlines()[:100]


# In[ ]:




