import torch.nn as nn
import torch

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

NUM_OF_EPOCHS = 1
BATCH_SIZE = 2
NUM_EPOCHS_TO_STAGNATE = 5

import shutil
total, used, free = shutil.disk_usage("/")
print("Total: %d GiB" % (total // (2**30)))
print("Used: %d GiB" % (used // (2**30)))
print("Free: %d GiB" % (free // (2**30)))

import sys

global start
start = time.time()

sys.path.append('../../../Parsing/')
print(os.getcwd())
from language_elements import Sentence, Word, Chunk

BERT_MODEL_NAMES = [
    'bert-base-uncased',
    'bert-large-uncased',
    'bert-base-multilingual-cased',
    'bert-large-uncased-whole-word-masking',  
]

with open('configuration.json', 'r') as json_file:
    config = json.load(json_file)

BATCH_SIZE = config['batch_size']
NUM_OF_EPOCHS = config['epochs']
NUM_EPOCHS_TO_STAGNATE = config['epochs_stagnate']
BERT_MODEL = config['bert-model-name']

with open('../../../Parsing/full_dataset_113.pickle', 'rb') as file:
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
print(list(index2words.keys()))

emb_list = []
def emb(word2index):
    with open('../../../Parsing/Embeddings/embeddings_dict_10_000.pickle', 'rb') as f:
        emb_dict = pickle.load(f)
    
    for e in emb_dict:
        print(e, emb_dict[e])
        break
    
    index2words_list = [(key, val) for key, val in index2words.items()]
    index2words_list = sorted(index2words_list)

    for index, word in index2words_list:
        if(word in emb_dict):
            emb_list.append(emb_dict[word])
        else:
            emb_list.append(np.random.rand(300))

train_data, test_data = train_test_split(tagged_sentences, test_size=0.1)
print("First sentence of train data:", train_data[0])
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

        return words, tags, is_heads, tags, y, seqlen

train_dataset = PosDataset(train_data)
eval_dataset = PosDataset(test_data)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 num_hidden_layers,
                 num_attention_heads,
                 intermediate_size,
                 hidden_act,
                 hidden_dropout_prob,
                 attention_probs_dropout_prob,
                 max_position_embeddings,
                 type_vocab_size,
                 initializer_range):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

class KannadaLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        super(KannadaLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config["hidden_size"]))
        self.beta = nn.Parameter(torch.zeros(config["hidden_size"]))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class KannadaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(KannadaEmbeddings, self).__init__()
        # self.word_embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.word_embeddings = nn.Embedding.from_pretrained(torch.Tensor(emb_list))

        self.LayerNorm = KannadaLayerNorm(config)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        # position_embeddings = self.position_embeddings(position_ids)
        # token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = words_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class KannadaSelfAttention(nn.Module):
    def __init__(self, config):
        super(KannadaSelfAttention, self).__init__()
        if config["hidden_size"] % config["num_attention_heads"] != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config["hidden_size"], config["num_attention_heads"]))
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = int(config["hidden_size"] / config["num_attention_heads"])
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config["hidden_size"], self.all_head_size)
        self.key = nn.Linear(config["hidden_size"], self.all_head_size)
        self.value = nn.Linear(config["hidden_size"], self.all_head_size)

        self.dropout = nn.Dropout(config["attention_probs_dropout_prob"])

class KannadaSelfOutput(nn.Module):
    def __init__(self, config):
        super(KannadaSelfOutput, self).__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.LayerNorm = KannadaLayerNorm(config)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class KannadaAttention(nn.Module):
    def __init__(self, config):
        super(KannadaAttention, self).__init__()
        self.self = KannadaSelfAttention(config)
        self.output = KannadaSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class KanandaIntermediate(nn.Module):
    def __init__(self, config):
        super(KanandaIntermediate, self).__init__()
        self.dense = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.intermediate_act_fn = ACT2FN[config["hidden_act"]] \
            if isinstance(config["hidden_act"], str) else config["hidden_act"]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class KannadaOutput(nn.Module):
    def __init__(self, config):
        super(KannadaOutput, self).__init__()
        self.dense = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.LayerNorm = KannadaLayerNorm(config)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class KannadaLayer(nn.Module):
    def __init__(self, config):
        super(KannadaLayer, self).__init__()
        self.attention = KannadaAttention(config)
        self.intermediate = KanandaIntermediate(config)
        self.output = KannadaOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class KannadaEncoder(nn.Module):
    def __init__(self, config):
        super(KannadaEncoder, self).__init__()
        layer = KannadaLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config["num_hidden_layers"])])    

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class KannadaPooler(nn.Module):
    def __init__(self, config):
        super(KannadaPooler, self).__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class KannadaBERT(nn.Module):
    def __init__(self, config):
        self.config = config
        super(KannadaBERT, self).__init__()
        self.embeddings = KannadaEmbeddings(config)
        self.encoder = KannadaEncoder(config)
        self.pooler = KannadaPooler(config)
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config["initializer_range"])
        elif isinstance(module, KannadaLayerNorm):
            module.beta.data.normal_(mean=0.0, std=self.config["initializer_range"])
            module.gamma.data.normal_(mean=0.0, std=self.config["initializer_range"])
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        print(type(encoded_layers))
        print(encoded_layers.shape)
        return encoded_layers, pooled_output


config = {
    "vocab_size": 20_000,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "initializer_range": 0.02   
}


model = KannadaBERT(config)
model = nn.DataParallel(model)
print(model)

def runner():
    def pad(batch):
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

    train_iter = torch.utils.data.DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                collate_fn=pad)

    test_iter = torch.utils.data.DataLoader(dataset=eval_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                collate_fn=pad)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    def train(model, iterator, optimizer, criterion):
        model.train()
        best_loss = None
        for eee in range(NUM_OF_EPOCHS):
            start_epoch = time.perf_counter()
            for i, batch in enumerate(iterator):
                words, x, is_heads, tags, y, seqlens = batch
                _y = y
                optimizer.zero_grad()
                logits, y, _ = model(x, y)

                logits = logits.view(-1, logits.shape[-1])
                y = y.view(-1)

                loss = criterion(logits, y)
                loss.backward()

                optimizer.step()

                if i%100==0:
                    global start
                    print("step: {}, loss: {:.2f}, time: {:.2f}s".format(i, loss.item(), time.time()-start))
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
                
            print(f"Epoch {eee+1} took {time.time()-start_epoch} time.")
            eval(model, test_iter)

            start_epoch = time.time()

    train(model, train_iter, optimizer, criterion)

runner()