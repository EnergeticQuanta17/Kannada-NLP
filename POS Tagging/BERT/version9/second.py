import torch.nn as nn
import torch
torch.set_default_dtype(torch.float64)

from sklearn.model_selection import train_test_split

import numpy as np
import math
import time

import json
import copy
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from transformers import BertTokenizer, BertModel

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

torch.manual_seed(42)

NUM_OF_EPOCHS = 1000
BATCH_SIZE = 1
NUM_EPOCHS_TO_STAGNATE = 100

import shutil
total, used, free = shutil.disk_usage("/")
print("Total: %d GiB" % (total // (2**30)))
print("Used: %d GiB" % (used // (2**30)))
print("Free: %d GiB" % (free // (2**30)))

class NotAllOneException(Exception):
    pass

class DebuggingTillHereException(Exception):
    pass

import sys

global start
start = time.time()

from language_elements import Sentence, Word, Chunk

BERT_MODEL_NAMES = [
    'bert-base-uncased',
    'bert-large-uncased',
    'bert-base-multilingual-cased',
    'bert-large-uncased-whole-word-masking',  
]

GPU_OR_COLAB = int(input("GPU(0) or COLAB(1): "))

if(GPU_OR_COLAB==0):
    CONFIG_PATH = "configuration.json"
else:
    CONFIG_PATH = "Kannada-NLP/POS Tagging/BERT/version6/configuration.json"

with open(CONFIG_PATH, 'r') as json_file:
    config = json.load(json_file)

BATCH_SIZE = config['batch_size']
NUM_OF_EPOCHS = config['epochs']
NUM_EPOCHS_TO_STAGNATE = config['epochs_stagnate']
BERT_MODEL = config['bert-model-name']
DATASET_PATH = config['dataset-path'][GPU_OR_COLAB]
EMBEDDINGS_PATH = config['embedding-path'][GPU_OR_COLAB]
CONFIG_PATH = config['configuration-path'][GPU_OR_COLAB]

with open(DATASET_PATH, 'rb') as file:
    retrieved_sentences = pickle.load(file)

tagged_sentences = []
for sentence in retrieved_sentences:
    temp = []
    for chunk in sentence.list_of_chunks:
        for word in chunk.list_of_words:
            temp.append((word.kannada_word, word.pos))
    tagged_sentences.append(temp)

all_words = list(set(word_pos[0] for sentence in tagged_sentences for word_pos in sentence))
tags = list(set(word_pos[1] for sentence in tagged_sentences for word_pos in sentence))
tags = ["<pad>"] + tags
NO_OF_TAGS = len(tags) - 1

tag2index = {tag:idx for idx, tag in enumerate(tags)}
index2tag = {idx:tag for idx, tag in enumerate(tags)}

words2index = {tag:idx for idx, tag in enumerate(all_words)}
index2words = {idx:tag for idx, tag in enumerate(all_words)}
# print(list(index2words.keys()))


def emb():
    emb_list_temp = []
    with open(EMBEDDINGS_PATH, 'rb') as f:
        emb_dict = pickle.load(f)
    
    index2words_list = [(key, val) for key, val in index2words.items()]
    index2words_list = sorted(index2words_list)

    for index, word in index2words_list:
        if(word in emb_dict):
            emb_list_temp.append(emb_dict[word])
        else:
            emb_list_temp.append(np.random.rand(300))
    
    return emb_list_temp

emb_list = emb()

train_data, test_data = train_test_split(tagged_sentences, test_size=0.1)
print("No. of sentences in train data:", len(train_data), "\nNo. of sentences in test data:", len(test_data))
print("Batch Size: ", BATCH_SIZE)
print("Number of EPOCHS:", NUM_OF_EPOCHS)

tokenizer = BertTokenizer.from_pretrained("l3cube-pune/kannada-bert")

model_name = "l3cube-pune/kannada-bert"
model = BertModel.from_pretrained(model_name)

class POSNet(nn.Module):
    def __init__(self, vocab_size=None):
        super().__init__()
        self.bert = model
        self.dropout = nn.Dropout(0.05)
        self.fc1 = nn.Linear(300, 256)
        self.fc2 = nn.Linear(256, vocab_size)
        self.device = device
    
    def forward(self, input_ids, y):
        input_ids = input_ids.to(self.device)
        y = y.to(self.device)
        
        if self.training:
            self.bert.train()
            
            attention_mask = torch.ones_like(input_ids)
            print(type(attention_mask))
            token_type_ids = torch.zeros_like(input_ids)
            
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            
            encoder_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            encoder_output = encoder_output.last_hidden_state
            
            print(encoder_output.shape)
            
            enc = encoder_output[-1]
            
            print(enc.shape)
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
                enc = encoded_layers[-1]
        
        enc = self.dropout(enc)
        enc = self.fc1(enc)
        logits = self.fc2(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat

text = "ನನಗೆ ಒಂದು ಪ್ರಶ್ನೆ ಇದೆ"
tokens = tokenizer.tokenize(text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)



# convert all of them to input_ids and then send to pos_model

pos_model = POSNet(vocab_size=len(tag2index))
pos_model.to(device)
model = nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

criterion = nn.CrossEntropyLoss(ignore_index=0)

print(tagged_sentences[0])

for sentence in tagged_sentences:
    optimizer.zero_grad()
    
    words, tags = zip(*sentence)
    
    tags = [tag2index[tag] for tag in tags]
    tags = torch.tensor([tags])
    
    
    tokens = tokenizer.tokenize(" ".join(words))
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([input_ids])
    
    print("Shape of input_ids: --> ", input_ids.shape)
    print("Shape of tags: --> ", tags.shape)
    
    logits, y, dk = pos_model(input_ids, tags)
    
    loss = criterion(logits, y)
    loss.backward()
    
    optimizer.step()
    scheduler.step()
    
    print(loss)
    break
    

# outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

