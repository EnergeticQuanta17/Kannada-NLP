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
# print("First sentence of train data:", train_data[0])
print("No. of sentences in train data:", len(train_data), "\nNo. of sentences in test data:", len(test_data))
print("Batch Size: ", BATCH_SIZE)
print("Number of EPOCHS:", NUM_OF_EPOCHS)
print("BERT Model used: ", BERT_MODEL)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
class PosDataset(torch.utils.data.Dataset):
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
            if(w in words2index):
                token_ids = [words2index[w]]
            else:
                token_ids = [0]
            is_head = [1]
            t = [t]

        
            y_ids = [tag2index[i] for i in t]

            x.extend(token_ids)
            is_heads.extend(is_head)
            y.extend(y_ids)
        try:
            assert len(x)==len(y)==len(is_heads)
        except:
            print(len(x), len(y), len(is_heads))

        seqlen = len(y)

        words = " ".join(words)
        tags = " ".join(tags)

        return words, x, is_heads, tags, y, seqlen

train_dataset = PosDataset(train_data)
eval_dataset = PosDataset(test_data)



from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("l3cube-pune/kannada-bert")
text = "ನನಗೆ ಒಂದು ಪ್ರಶ್ನೆ ಇದೆ"

tokens = tokenizer.tokenize(text)
print(tokens)

input_ids = tokenizer.convert_tokens_to_ids(tokens)

import torch
import torch.nn as nn

input_ids = torch.tensor([input_ids])
attention_mask = torch.ones_like(input_ids)
token_type_ids = torch.zeros_like(input_ids)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
token_type_ids = token_type_ids.to(device)

model_name = "l3cube-pune/kannada-bert"
model = BertModel.from_pretrained(model_name)

model.to(device)
outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

print(outputs.last_hidden_state)


class POSNet(nn.Module):
    def __init__(self, vocab_size=None):
        super().__init__()
        self.bert = model
        self.dropout = nn.Dropout(0.05)
        self.fc1 = nn.Linear(300, 256)
        self.fc2 = nn.Linear(256, vocab_size)
        self.device = device
    
    def forward(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)

        # print(x)
        # print(x.shape)
        
        if self.training:
            self.bert.train()
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
                enc = encoded_layers[-1]
        
        # enc = self.dropout(enc)
        enc = self.fc1(enc)
        logits = self.fc2(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat
    
    

def runner():
    def pad(batch):
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

    train_iter = torch.utils.data.DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                collate_fn=pad)

    test_iter = torch.utils.data.DataLoader(dataset=eval_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                collate_fn=pad)

    model = POSNet(vocab_size=len(tag2index))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model = nn.DataParallel(model)

    print(model)
    print(model.parameters())
    print("For These parmaters, requires_grad is not done")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print("DOES NOT REQUIRE --> ", name)
        else:
            # print("REQUIRES --> ", name)
            pass

    from torchvision import models
    from torchsummary import summary

    
    vgg = models.vgg16().to(device)
    # summary(vgg, (3, 224, 224))

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    def train(model, iterator, optimizer, scheduler, criterion):
        model.train()
        best_loss = None
        for eee in range(NUM_OF_EPOCHS):
            start_epoch = time.perf_counter()
            for i, batch in enumerate(iterator):
                words, x, is_heads, tags, y, seqlens = batch
                _y = y
                print(f"Shape of x: {x.shape}")
                print(f"Shape of y: {y.shape}")
                optimizer.zero_grad()
                logits, y, dk = model(x, y)
                print("After going to model:", f"Logits: {logits.shape}", f"y: {y.shape}", f"DK: {dk.shape}")
                print(f"Shape of logits of logits.shape[-1]: {logits.shape[-1]}")
                logits = logits.view(-1, logits.shape[-1])
                print("Shape of logits after transforming: ", logits.shape)

                y = y.view(-1)

                print(logits)
                print()
                print(y)
                print()
                print(logits.shape, y.shape)
                print()
                print(logits[0])
                print()
                print("Argmax for first: ", y[0])
                print()
                agmax = torch.argmax(logits[0]).item()
                print("Argmax in logits index: ", torch.argmax(logits[0]).item())
                print()
                print("Element in that positon: ", logits[0][agmax])
                print()
                # print("Loss for first index:", criterion(logits[0], y))
                print()
                probs = torch.exp(logits) / torch.exp(logits).sum()
                print("Probs vector", probs)                
                lossi = -torch.sum(torch.log(probs) * y)
                print("Loss on sinde index from scratch", lossi)
                loss = criterion(logits, y)

                raise Exception
                
                before_update = {}
                for name, param in model.named_parameters():
                    before_update[name] = param.clone()

                # Backward pass
                loss.backward()

                # Update the model parameters
                optimizer.step()
                scheduler.step()

                # After optimizer.step()
                after_update = {}
                for name, param in model.named_parameters():
                    after_update[name] = param.clone()

                # Check which parameters were updated
                # updated_params = []
                # for name in before_update:
                #     if not torch.equal(before_update[name], after_update[name]):
                #         updated_params.append(name)
                #         if(name not in ["module.bert.embeddings.word_embeddings.weight", "module.bert.pooler.dense.bias", "module.bert.pooler.dense.weight"]):
                #             print(name)
                #             for named, param in model.named_parameters():
                #                 if(name==named):
                #                     print("Gradient change:", param.grad.data.norm())
                        # raise DebuggingTillHereException
                

                # for name, param in model.named_parameters():
                #     if(name == "module.bert.encoder.layer.0.attention.self.query.weight"):
                #         # print("Parameter name:", name)
                #         print("Gradient change:", param.grad.data.norm())
                #         # print("Parameter value:", param, "\n--------------------------------------------")
                #         # print()
                #     elif(name == "module.bert.encoder.layer.11.output.dense.weight"):
                #         # print("Parameter name:", name)
                #         print("Gradient change:", param.grad.data.norm())
                #         # print("Parameter value:", param, "\n--------------------------------------------")
                #         # print()           
                #     elif(name == "module.fc2.weight"):
                #         # print("Parameter name:", name)
                #         print("Gradient change:", param.grad.data.norm())
                #         print(len(updated_params), len(before_update))
                #         # print("Parameter value:", param, "\n=============================================\n")
                #         print("\n=============================================\n")
                #         print()   
                    

                if i%100==0:
                    global start
                    print("step: {}, loss: {:.2f}, time: {:.2f}s".format(i, loss.item(), time.time()-start))
                    eval(model, train_iter)
                    start = time.time()
                    
                    if best_loss is None or loss < best_loss:
                        best_loss = loss    
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                    if epochs_without_improvement >= NUM_EPOCHS_TO_STAGNATE:
                        print('-----------------------------')
                        print("Loss has become stagnant.")
                        print('-----------------------------')
                        break
                
            print(f"Epoch {eee+1} took {time.perf_counter()-start_epoch} time.")
            eval(model, train_iter)
            eval(model, test_iter)

            start_epoch = time.time()

    def eval(model, iterator):
        model.eval()

        Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                words, x, is_heads, tags, y, seqlens = batch

                _, _, y_hat = model(x, y)

                Words.extend(words)
                Is_heads.extend(is_heads)
                Tags.extend(tags)
                Y.extend(y.numpy().tolist())
                Y_hat.extend(y_hat.cpu().numpy().tolist())

        with open("result", 'w') as fout:
            count = 0
            for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
                for each in is_heads:
                    if(each!=1):
                        raise NotAllOneException
                y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
                
                preds = [index2tag[hat] for hat in y_hat]
                try:
                    assert len(preds)==len(words.split())==len(tags.split())
                    for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                        fout.write("{} {} {}\n".format(w, t, p))
                        fout.write("\n")
                except:
                    print("Assertion Error: ", preds, words, tags)

                

        # print(index2tag)

        y_true =  np.array([tag2index[line.split()[1]] for line in open('result', 'r').read().splitlines() if len(line) > 0])
        y_pred =  np.array([tag2index[line.split()[2]] for line in open('result', 'r').read().splitlines() if len(line) > 0])

        acc = (y_true==y_pred).astype(np.int32).sum() / len(y_true)

        print("acc=%.4f"%acc)

    train(model, train_iter, optimizer, scheduler, criterion)
    eval(model, test_iter)

runner()

open('result', 'r').read().splitlines()[:100]




# file_path = "/home/preetham/Results/BERT/version6/results.json"
# if not os.path.exists("/home/preetham/Results/BERT/version6"):
#     os.makedirs("path/to/demo_folder")

# if os.path.exists(file_path):
#     with open(file_path, "r") as file:
#         json_data = json.load(file)
# else:
#     json_data = []

# json_data.append(data)

# with open(file_path, "w") as file:
#     json.dump(json_data, file, indent=4)
