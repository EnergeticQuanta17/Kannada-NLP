import json
import numpy as np
import pickle, sys, os
import tensorflow as tf

from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, Bidirectional
from keras.models import Model
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

tf.get_logger().setLevel('ERROR')

class DebuggingTillHereException(Exception):
    pass

IDEA_NUMBER = 3
NO_OF_EMBEDDINGS = 10000
MAX_SEQUENCE_LENGTH = 140
EMBEDDING_DIM = 300
TEST_SPLIT = 0.05
VALIDATION_SPLIT = 0.05
BATCH_SIZE = 64
UNITS_IN_LSTM_LAYER = 64
EPOCHS = 15

with open('all_data.pkl', 'rb') as f:
    X, y, word2int, int2word, tag2int, int2tag = pickle.load(f)


# Tag 0 for the PAD
int2tag[0] = '<PAD>'
tag2int['<PAD>'] = 0
n_tags = len(tag2int)

# Data Size
print('\n\n')
print("Shape of X : ", len(X))
print("Shape of Y : ", len(y))
print("Shape of word2int :", len(word2int))
print("Shape of int2word :", len(int2word))
print("Shape of tag2int :", len(tag2int))
print("Shape of int2tag :", len(int2tag))
print('\n\n')


# Data Samples
print("X : ", X[:5])
print("Y : ", y[:5])
print("word2int :", word2int[:5])
print("int2word :", int2word[:5])
print("tag2int :", tag2int[:5])
print("int2tag :", int2tag[:5])
print('\n\n')


# Generator for Batch Processing
def generator(all_X, all_y, n_classes, batch_size=BATCH_SIZE):
    num_samples = len(all_X)
    while True:
        for offset in range(0, num_samples, batch_size):
            X = all_X[offset:offset+batch_size]
            y = all_y[offset:offset+batch_size]
            y = to_categorical(y, num_classes=n_classes)
            yield shuffle(X, y)


X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
y = pad_sequences(y, maxlen=MAX_SEQUENCE_LENGTH)


# Total TAGS & WORDS
print('TOTAL TAGS ', len(tag2int))
print('TOTAL WORDS ', len(word2int))


# shuffle the data
X, y = shuffle(X, y)

# Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT,random_state=42)

# Validation Split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VALIDATION_SPLIT, random_state=1)

n_train_samples = X_train.shape[0]
n_val_samples = X_val.shape[0]
n_test_samples = X_test.shape[0]
print('TRAINING samples ', n_train_samples)
print('VALIDATION samples ', n_val_samples)
print('TEST samples ', n_test_samples)


# Generators for training and validation
train_generator = generator(all_X=X_train, all_y=y_train, n_classes=n_tags)
validation_generator = generator(all_X=X_val, all_y=y_val, n_classes=n_tags)


# Embeddings
with open('../../../Parsing/Embeddings/embeddings_dict_10_000.pickle', 'rb') as f:
	embeddings_index = pickle.load(f)

print('Total %s word vectors.' % len(embeddings_index))

# + 1 to include the unkown word
embedding_matrix = np.random.random((len(word2int) + 1, EMBEDDING_DIM))

word_not_in_embedding = 0

for word, i in word2int.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        word_not_in_embedding += 1
        # words not found in embeddings_index will remain unchanged and thus will be random.
        embedding_matrix[i] = embedding_vector

print('Word not in Embedding : ', word_not_in_embedding)
print('Embedding matrix shape', embedding_matrix.shape)
print('X_train shape', X_train.shape)


# Embedding + BiLSTM Layer
embedding_layer = Embedding(len(word2int)+1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_lstm = Bidirectional(LSTM(UNITS_IN_LSTM_LAYER, return_sequences=True), merge_mode='sum')(embedded_sequences)
preds = TimeDistributed(Dense(n_tags, activation='softmax'))(l_lstm)

weightFile = 'weightFile' + '-{epoch:d}-{loss:.2f}.wts'
checkpointCallback = ModelCheckpoint(weightFile, monitor='val_loss', verbose=0,
                                    save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

model = Model(sequence_input, preds)

model.compile(loss='SparseCategoricalCrossentropy',
              optimizer='adam',
              metrics=['accuracy'])


print("model fitting - Bidirectional LSTM")
model.summary()

history  = model.fit(train_generator, 
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

y_test = to_categorical(y_test, num_classes=n_tags)
test_results = model.evaluate(X_test, y_test, verbose=0)

X_test

print('TEST LOSS %f \nTEST ACCURACY: %f' % (test_results[0], test_results[1]))

print("X_test: ", X_test.shape)
y_pred = model.predict(X_test)


print('\n\n\n-------------------------------------------------------------------------------------------------------------------------------------------\n\n')
print(int2tag)

print('X_test shape ', X_test.shape)
print('Y_pred shape ' , y_pred.shape)

y_pred = y_pred.reshape(-1, 77)
y_test = y_test.reshape(-1, 77)


# print('Y_pred shape ' , y_pred.shape)

print()
print()

"""
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
            for index, y in enumerate(zip(y_pred[k], y_test[k])):
                print(index, '-->', y[0], '\t', y[1])
                print(int2tag[index])
                
            count+=1
            break
    

        with open('find_this_sentence.pkl', 'wb') as file:
            pickle.dump(X_test[k], file)

        
        # print("Printing argmax tag of predict", int2tag[np.argmax(y_pred[k])])
        # print("Printing argmax tag of test", int2tag[np.argmax(y_test[k])])
        
        

        # for i, j in zip(y_pred[k], y_test[k]):
        #     print(i, '\t', j)

print("Number of non-zero indexed tags: ", count)

"""

with open('index_to_word.txt', 'w') as f:
    for i in int2word:
        f.write(str(i) + '-->' + int2word[i] + '\n')
    
with open('index_to_tag.txt', 'w') as f:
    for i in int2tag:
        f.write(str(i) + '-->' + int2tag[i] + '\n')

count = 0
non_zero_count = 0
tcount = 0
tcount_zeros = 0

if(True):
    for k in range(y_pred.shape[0]):
        if(np.argmax(y_pred[k]) == np.argmax(y_test[k])):
            if(np.argmax(y_test[k])!=0):
                #for i, j in zip(y_pred[k], y_test[k]):
                    #print(i, '\t', j)
                non_zero_count += 1
            count+=1
        if(np.argmax(y_test[k])!=0):
            tcount += 1
        tcount_zeros += 1

print()
print('Non zero count : ', non_zero_count)
print('Total count : ', tcount)
print("Total count including zeros:", tcount_zeros)
ACCURACY = 100* non_zero_count/tcount
print("Accuracy: ", ACCURACY)




# LOADING TO JSON (NOT ACCURATE)

# data = {
#     "Idea Number": IDEA_NUMBER,
#     "Epochs": EPOCHS,
#     "Number of Embeddings used": NO_OF_EMBEDDINGS,
#     "Max Sequence Length": MAX_SEQUENCE_LENGTH,
#     "Embedding Dimension": EMBEDDING_DIM,
#     "Test split": TEST_SPLIT,
#     "Validation split": VALIDATION_SPLIT,
#     "Number of train samples": n_train_samples,
#     "Number of validation samples": n_val_samples,
#     "Number of test samples": n_test_samples,
#     "Batch Size": BATCH_SIZE,
#     "Number of units in LSTM Layer": UNITS_IN_LSTM_LAYER,
#     "Embeddings File" : "Embeddings/embeddings_dict_10_000.pickle",
#     "Test Loss": test_results[0],
#     "Test Accuracy": test_results[1],
#     "Inflated Training Accuracy": training_accuracy,
#     "Inflated Train Loss":history.history['loss'],
#     "Inflated Validation Accuracy": history.history['val_acc'],
#     "Inflated Validation Loss":history.history['val_loss'],
#     "Number of parametrs": int(input("Enter the number of parameters in this model:")),
#     'Accuracy': ACCURACY
# }

# file_path = "results.json" 

# if os.path.exists(file_path):
#     with open(file_path, "r") as file:
#         json_data = json.load(file)
# else:
#     json_data = []

# model.save(f'Models/model{len(json_data)}.h5')

# json_data.append(data)

# with open(file_path, "w") as file:
#     json.dump(json_data, file, indent=4)