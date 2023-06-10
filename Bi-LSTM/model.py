import json
import numpy as np
import pickle, sys, os

from keras_preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, Bidirectional
from keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

NO_OF_EMBEDDINGS = 10000
MAX_SEQUENCE_LENGTH = 140
EMBEDDING_DIM = 300
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32
UNITS_IN_LSTM_LAYER = 64
EPOCHS = 25


with open('all_data.pkl', 'rb') as f:
    X, y, word2int, int2word, tag2int, int2tag = pickle.load(f)

print("Shape of X: ", X.shape)
print("Shape of Y: ", y.shape)


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

# y = to_categorical(y, num_classes=len(tag2int) + 1)

print('TOTAL TAGS', len(tag2int))
print('TOTAL WORDS', len(word2int))

# shuffle the data
X, y = shuffle(X, y)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT,random_state=42)

# split training data into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VALIDATION_SPLIT, random_state=1)

n_train_samples = X_train.shape[0]
n_val_samples = X_val.shape[0]
n_test_samples = X_test.shape[0]

print('We have %d TRAINING samples' % n_train_samples)
print('We have %d VALIDATION samples' % n_val_samples)
print('We have %d TEST samples' % n_test_samples)

# make generators for training and validation
train_generator = generator(all_X=X_train, all_y=y_train, n_classes=n_tags + 1)
validation_generator = generator(all_X=X_val, all_y=y_val, n_classes=n_tags + 1)



with open('Embeddings/embeddings_dict_10_000.pickle', 'rb') as f:
	embeddings_index = pickle.load(f)

print('Total %s word vectors.' % len(embeddings_index))

# + 1 to include the unkown word
embedding_matrix = np.random.random((len(word2int) + 1, EMBEDDING_DIM))

for word, i in word2int.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embeddings_index will remain unchanged and thus will be random.
        embedding_matrix[i] = embedding_vector

print('Embedding matrix shape', embedding_matrix.shape)
print('X_train shape', X_train.shape)

embedding_layer = Embedding(len(word2int)+1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

l_lstm = Bidirectional(LSTM(UNITS_IN_LSTM_LAYER, return_sequences=True))(embedded_sequences)
preds = TimeDistributed(Dense(n_tags + 1, activation='softmax'))(l_lstm)
model = Model(sequence_input, preds)


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Bidirectional LSTM")
model.summary()

history  = model.fit_generator(train_generator, 
                     steps_per_epoch=n_train_samples//BATCH_SIZE,
                     validation_data=validation_generator,
                     validation_steps=n_val_samples//BATCH_SIZE,
                     epochs=EPOCHS,
                     verbose=1,)

training_accuracy = history.history['acc']
print("Training Accuracy:", training_accuracy)

if not os.path.exists('Models/'):
    print('MAKING DIRECTORY Models/ to save model file')
    os.makedirs('Models/')

train = True

# if train:
#     model.save('Models/model.h5')
#     print('MODEL SAVED in Models/ as model.h5')
# else:
#     from keras.models import load_model
#     model = load_model('Models/model.h5')

y_test = to_categorical(y_test, num_classes=n_tags+1)
test_results = model.evaluate(X_test, y_test, verbose=0)

X_test

print('TEST LOSS %f \nTEST ACCURACY: %f' % (test_results[0], test_results[1]))

print("X_test: ", X_test.shape)
y_pred = model.predict(X_test)

print("Type of y-pred: ", type(y_pred), y_pred.shape)

# print()
# print(y_pred[:20])

for y in y_pred[:20]:
    print(np.argmax(y))


print("Type of y-test: ", type(y_test), y_test.shape)

# print(y_test[:20])
for y in y_test[:20]:
    print(np.argmax(y))



print('\n\n\n-------------------------------------------------------------------------------------------------------------------------------------------\n\n')



y_pred = y_pred.reshape(-1, 73)
y_test = y_test.reshape(-1, 73)

count = 0
if(True):
    for k in range(X_test.shape[0]):
        print(np.argmax(y_pred[k]))
        # print('Word vector -->')
        # print(X_test[k])
        # print()
        # for i, j in zip(y_pred[k], y_test[k]):
        #     print(i, '\t', j)
        if(np.argmax(y_test[k]) != 0):
            # for index, y in enumerate(zip(y_pred[k], y_test[k])):
            #     print(index, '-->', y[0], '\t', y[1])
            #     print(int2tag[index])
            count+=1
    

        with open('find_this_sentence.pkl', 'wb') as file:
            pickle.dump(X_test[k], file)
        print("Printing argmax tag of predict", int2tag[np.argmax(y_pred[k])+1])
        print("Printing argmax tag of test", int2tag[np.argmax(y_test[k])+1])
        
        

        # for i, j in zip(y_pred[k], y_test[k]):
        #     print(i, '\t', j)

print("Number of non-zero indexed tags: ", count)

with open('index_to_word.txt', 'w') as f:
    for i in int2word:
        f.write(str(i) + '-->' + int2word[i] + '\n')
    
with open('index_to_tag.txt', 'w') as f:
    for i in int2tag:
        f.write(str(i) + '-->' + int2tag[i] + '\n')

count = 0
if(True):
    for k in range(X_test.shape[0]):
        if(np.argmax(y_pred[k]) == np.argmax(y_test[k])):
            # if(np.argmax(y_pred[k])!=0):
            #     for i, j in zip(y_pred[k], y_test[k]):
            #         print(i, '\t', j)
            #     break
            count+=1
print()
print()
print(count, X_test.shape[0])
ACCURACY = 100* count/X_test.shape[0]
print("Accuracy: ", ACCURACY)

# for i, j in zip(y_pred[10], y_test[10]):
#     print(i, '\t', j)


data = {
    "Epochs": EPOCHS,
    "Number of Embeddings used": NO_OF_EMBEDDINGS,
    "Max Sequence Length": MAX_SEQUENCE_LENGTH,
    "Embedding Dimension": EMBEDDING_DIM,
    "Test split": TEST_SPLIT,
    "Validation split": VALIDATION_SPLIT,
    "Number of train samples": n_train_samples,
    "Number of validation samples": n_val_samples,
    "Number of test samples": n_test_samples,
    "Batch Size": BATCH_SIZE,
    "Number of units in LSTM Layer": UNITS_IN_LSTM_LAYER,
    "Embeddings File" : "Embeddings/embeddings_dict_10_000.pickle",
    "Test Loss": test_results[0],
    "Test Accuracy": test_results[1],
    "Training Accuracy": training_accuracy,
    'Accuracy': ACCURACY
}

file_path = "../../results.json" 

if os.path.exists(file_path):
    with open(file_path, "r") as file:
        json_data = json.load(file)
else:
    json_data = []

model.save(f'Models/model{len(json_data)}.h5')

json_data.append(data)

with open(file_path, "w") as file:
    json.dump(json_data, file, indent=4)