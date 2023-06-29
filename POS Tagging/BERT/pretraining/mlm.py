import torch
import random
from transformers import BertTokenizer

def tokenizer(sentence):
    

def masked_language_modeling(sentence, tokenizer, mask_rate=0.15):
    masked_tokens = []
    for token in tokenizer(sentence):
        print(type(token))
        if random.random() < mask_rate:
            masked_tokens.append("[MASK]")
        else:
            masked_tokens.append(token)
    return sentence, masked_tokens

if __name__ == "__main__":
    sentence = 'ನಿಮ್ಮ ಹೆಸರು ಏನು?'
    
    masked_sentence, masked_tokens = masked_language_modeling(sentence, tokenizer)
    print(masked_sentence)
    print(masked_tokens)
