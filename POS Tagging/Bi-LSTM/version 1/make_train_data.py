import pickle
import random
import os

import sys
sys.path.append('../Parsing/')

from language_elements import Sentence, Word, Chunk

# file_name = input('FileName: ')
file_name = "../Parsing/full_dataset_113.pickle"

print(os.getcwd())

# Split --> 0.8, 0.1, 0.1
train_size = 4334
validation_size = 542
test_size = 542

block1_start = random.randint(0, 5418 - validation_size)
block2_start = random.randint(0, 5418 - test_size)

while block2_start >= block1_start and block2_start <= block1_start + 542:
    block2_start = random.randint(0, 5418 - 542)

validation_range = range(block1_start, block1_start + 542)
test_range = range(block2_start, block2_start + 542)

print(block1_start)
print(block2_start)


full_dataset_sentences = []
with open('train.data', 'w', encoding='utf-8') as train_file:
    with open('validation.data', 'w', encoding='utf-8') as validation_file:
        with open('test.data', 'w', encoding='utf-8') as test_file:
            with open(file_name, 'rb') as file:
                retrieved_sentences = pickle.load(file)

                for index, sentence in enumerate(retrieved_sentences):
                    for chunk in sentence.list_of_chunks:
                        for word in chunk.list_of_words:
                            if(index in validation_range):
                                validation_file.write(word.kannada_word + '\t' + word.pos + '\t' + chunk.chunk_group + '\n')
                            elif(index in test_range):
                                test_file.write(word.kannada_word + '\t' + word.pos + '\t' + chunk.chunk_group + '\n')
                            else:
                                train_file.write(word.kannada_word + '\t' + word.pos + '\t' + chunk.chunk_group + '\n')
                    # print(sentence.id)

# print(len(retrieved_sentences))
