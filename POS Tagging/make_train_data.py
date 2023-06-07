import pickle

import sys
sys.path.append('../Parsing/')

from language_elements import Sentence, Word, Chunk

# file_name = input('FileName: ')
file_name = "full_dataset_113.pickle"

full_dataset_sentences = []
with open('train.data', 'w', encoding='utf-8') as train_file:
    with open(file_name, 'rb') as file:
        retrieved_sentences = pickle.load(file)

        for sentence in retrieved_sentences:
            for chunk in sentence.list_of_chunks:
                for word in chunk.list_of_words:
                    train_file.write(word.kannada_word + '\t' + word.pos + '\t' + chunk.chunk_group + '\n')
            print(sentence.id)

# print(len(retrieved_sentences))
