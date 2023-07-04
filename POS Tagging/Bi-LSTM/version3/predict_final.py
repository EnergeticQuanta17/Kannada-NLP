import numpy as np
import pickle
import time

from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.utils.np_utils import to_categorical

from sklearn.utils import shuffle

BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH = 140

model = load_model('Models/model.h5')

with open('all_data.pkl', 'rb') as f:
    X, y, word2int, int2word, tag2int, int2tag = pickle.load(f)

print("Printing the len of int2tag before:", len(int2tag))

int2word[0] = '<PAD>'
int2tag[0] = '<PAD>'
tag2int['<PAD>'] = 0

print("Printing the len of int2tag after:", len(int2tag))

print("Shape of X: ", len(X))
print("Shape of Y: ", len(y))

n_tags = len(tag2int)

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
y = pad_sequences(y, maxlen=MAX_SEQUENCE_LENGTH)

print(X.shape, y.shape)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

words, encoded_words = [], []
with open('input_words.txt', 'r') as f:
    for line in f:
        line = line.strip()
        words.append(line)
        encoded_words.append(word2int[line])

# print(words)
# print(encoded_words)

padded_words = pad_sequences([encoded_words], maxlen=MAX_SEQUENCE_LENGTH)
# print(padded_words)
# print()

# print(model.summary())
# print()

y_pred = model.predict(padded_words)
pred_0 = y_pred[0]


actual_tags = []
with open('input_tags.txt', 'r') as f:
    for line in f:
        actual_tags.append(line.strip())

for index, ele in enumerate(pred_0[-1 * len(words):]):
    

    if(int2tag[np.argmax(ele)] != actual_tags[index]):
        print(f"{words[index]:20s} {int2tag[np.argmax(ele)]:10s} {actual_tags[index]:10s}", end='   ')
        print("(x)")
        time.sleep(15)
        raise Exception
    else:
        print(f"{words[index]:20s} {int2tag[np.argmax(ele)]:10s} {actual_tags[index]:10s}", end='   ')

print()
print()
time.sleep(5)