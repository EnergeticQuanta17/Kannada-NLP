import numpy as np
import pickle
import time

from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model

BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH = 30

model = load_model('Models/model.h5')

with open('all_data.pkl', 'rb') as f:
    X, y, word2int, int2word, tag2int, int2tag = pickle.load(f)

int2word[0] = '<PAD>'
int2tag[0] = '<PAD>'
tag2int['<PAD>'] = 0

def tagger(input_words_file):
    position = []
    words, encoded_words, sep_words = [], [], []
    with open(input_words_file, 'r') as f:
        for index, line in enumerate(f):
            line = line.strip()
            if(line):
                words.append(line)
                if(line in word2int):
                    sep_words.append(line)
                    encoded_words.append(word2int[line])
                    position.append(-1)
                else:
                    position.append(index)

    padded_words = pad_sequences([encoded_words], maxlen=MAX_SEQUENCE_LENGTH)

    y_pred = model.predict(padded_words)
    pred_0 = y_pred[0]

    text, y_hat = [], []
    for index, ele in enumerate(pred_0[-1 * len(encoded_words):]):
        text.append(words[index])
        y_hat.append(int2tag[np.argmax(ele)])
    
    print(y_hat)

    tagset = list(tag2int.keys()) + ['UNK_NOUN']

    final_text, final_y_hat = [], []
    
    text_index = 0

    for i in position:
        if(i==-1):
            final_text.append(sep_words[text_index])
            final_y_hat.append(y_hat[text_index])
            text_index += 1
        else:
            final_text.append(words[i])
            final_y_hat.append('NOUN')
    
    return final_text, final_y_hat, tagset