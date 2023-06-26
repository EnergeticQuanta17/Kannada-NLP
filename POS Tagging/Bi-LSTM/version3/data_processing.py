import os
import sys
import numpy as np
import pickle

X_train = []
Y_train = []

words = []
tags = []

sys.path.append('../../../AnnotatedDatasetParsing/')
from language_elements import Sentence, Chunk, Word

with open('../../../AnnotatedDatasetParsing/full_dataset_131.pickle', 'rb') as file:
    retrieved_sentences = pickle.load(file)

print("Number of sentences retrieved: ", len(retrieved_sentences))

for sentence in retrieved_sentences:
    tempX = []
    tempY = []
    for chunk in sentence.list_of_chunks:
        for word in chunk.list_of_words:
            words.append(word.kannada_word)
            tags.append(word.pos)
            # print("Word ", word.kannada_word,)
            # print("tags ", word.pos)

            tempX.append(word.kannada_word)
            tempY.append(word.pos)
    
    X_train.append(tempX)
    Y_train.append(tempY)

lengths = []
for sentence in retrieved_sentences:
    temp = 0
    for chunk in sentence.list_of_chunks:
        temp += len(chunk.list_of_words)
    
    lengths.append(temp)

import statistics
print("Mean:", statistics.mean(lengths))
print("Median:", statistics.median(lengths))
print("Mode:", statistics.mode(lengths))
print("stdev:", statistics.stdev(lengths))



# for ele in X_train:
#   print(ele)
# print('X_train shape:', np.array(X_train).shape)
# print('Y_train shape:', np.array(Y_train).shape)

words = set(words)
tags = set(tags)



print("\n\n----------------------------PART OF SPEECH TAGS----------------------------")
for index, tag in enumerate(tags):
    print(index, '\t', tag)


print("\n\n----------------------------FIRST 10 WORDS----------------------------")
for word in list(words)[:10]:
    print(word)


print("\n\n--------------------------------------------------------------------")
print('Vocabulary Size: ', len(words))
print('Total POS Tags: ', len(tags))

# TOKEN TO EACH WORD

word2int = {}
int2word = {}

for i, word in enumerate(words):
    word2int[word] = i+1
    int2word[i+1] = word

tag2int = {}
int2tag = {}

# TOKEN TO EACH TAG

for i, tag in enumerate(tags):
    tag2int[tag] = i+1
    int2tag[i+1] = tag

    # if(tag=='10ರಲ್ಲಿ'):
    #     print("10ರಲ್ಲಿ Tag found")

# EMPTY TAG
# print("Seee this", int2tag[1])

X_train_numberised = []
Y_train_numberised = []

# FOR SENTENCES

for sentence in X_train:
    tempX = []
    for word in sentence:        
        tempX.append(word2int[word])
    X_train_numberised.append(tempX)

# FOR TAGS

for tags in Y_train:
    tempY = []
    for tag in tags:
        tempY.append(tag2int[tag])
    Y_train_numberised.append(tempY)

print()
print('sample X_train_numberised: ', X_train_numberised[42], '\n')
print('sample Y_train_numberised: ', Y_train_numberised[42], '\n')

# print('X_train_numberised shape:', np.array(X_train_numberised).shape)
# print('Y_train_numberised shape:', np.array(Y_train_numberised).shape)

# X_train_numberised = np.asarray(X_train_numberised)
# Y_train_numberised = np.asarray(Y_train_numberised)

pickle_files = [X_train_numberised, Y_train_numberised, word2int, int2word, tag2int, int2tag]

with open('all_data.pkl', 'wb') as f:
    pickle.dump(pickle_files, f)

print('Saved as pickle file')