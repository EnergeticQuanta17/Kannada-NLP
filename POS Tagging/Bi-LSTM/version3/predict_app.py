import numpy as np
import pickle
import time

from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model

BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH = 140

model = load_model('Models/model.h5')

with open('all_data.pkl', 'rb') as f:
    X, y, word2int, int2word, tag2int, int2tag = pickle.load(f)

int2word[0] = '<PAD>'
int2tag[0] = '<PAD>'
tag2int['<PAD>'] = 0

def tagger(input_words_file):
    words, encoded_words = [], []
    with open(input_words_file, 'r') as f:
        for line in f:
            line = line.strip()
            words.append(line)
            encoded_words.append(word2int[line])

    padded_words = pad_sequences([encoded_words], maxlen=MAX_SEQUENCE_LENGTH)

    y_pred = model.predict(padded_words)
    pred_0 = y_pred[0]

    text, y_hat = [] * 2
    for index, ele in enumerate(pred_0[-1 * len(words):]):
        text.append(words[index])
        y_hat.append(int2tag[np.argmax(ele)])

    tagset = list(tag2int.keys())

    return text, y_hat, tagset