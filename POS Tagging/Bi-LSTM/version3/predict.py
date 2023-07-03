import pickle

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


def generator(all_X, all_y, n_classes, batch_size=BATCH_SIZE):
    num_samples = len(all_X)

    while True:

        for offset in range(0, num_samples, batch_size):
            
            X = all_X[offset:offset+batch_size]
            y = all_y[offset:offset+batch_size]

            y = to_categorical(y, num_classes=n_classes)


            yield shuffle(X, y)


n_tags = len(tag2int)

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
y = pad_sequences(y, maxlen=MAX_SEQUENCE_LENGTH)

print(X.shape, y.shape)

words, encoded_words = [], []
with open('input.txt', 'r') as f:
    for line in f:
        line = line.strip()
        words.append(line)
        encoded_words.append(word2int[line])

print(words)
print(encoded_words)

padded_words = pad_sequences([encoded_words], maxlen=MAX_SEQUENCE_LENGTH)
print(padded_words)
print()

print(model.summary())
print()

y_pred = model.predict(padded_words)
print(len(y_pred))
print(len(y_pred[0]))
print(len(y_pred[0][0]))