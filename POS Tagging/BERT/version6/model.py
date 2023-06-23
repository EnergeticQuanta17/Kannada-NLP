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

# print("Printing the 26th index of train_dataset:", train_dataset[26])

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
        self.word_embeddings = nn.Embedding.from_pretrained(torch.tensor(emb_list, requires_grad=True))
        # self.word_embeddings = nn.Embedding.from_pretrained(torch.tensor(emb_list))

        self.LayerNorm = KannadaLayerNorm(config)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, requires_grad=True)

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

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

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
            print(type(encoded_layers))
            encoded_layers = encoded_layers[-1]

        # print("Length of encoded_layers:", len(encoded_layers))
        return encoded_layers, pooled_output


config = {
    "vocab_size": 20_000,
    "hidden_size": 300,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "initializer_range": 0.02   
}

# config = {
#     "vocab_size": 20_000,
#     "hidden_size": 300,
#     "num_hidden_layers": 30,
#     "num_attention_heads": 30,
#     "intermediate_size": 3072,
#     "hidden_act": "swish",
#     "hidden_dropout_prob": 0.01,
#     "attention_probs_dropout_prob": 0.01,
#     "max_position_embeddings": 0,
#     "type_vocab_size": 0,
#     "initializer_range": 0.02   
# }



# model = KannadaBERT(config)
# model = nn.DataParallel(model)
# print(model)

class POSNet(nn.Module):
    def __init__(self, vocab_size=None):
        super().__init__()
        self.bert = KannadaBERT(config)
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
                print("Loss for first index:", criterion(logits[0], y))
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
