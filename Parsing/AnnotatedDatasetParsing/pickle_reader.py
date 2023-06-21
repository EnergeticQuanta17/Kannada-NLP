import pickle

from language_elements import Sentence, Word, Chunk

file_name = input('FileName: ')

full_dataset_sentences = []
with open(file_name, 'rb') as file:
    retrieved_sentences = pickle.load(file)
    for sentence in retrieved_sentences:
        print(sentence)

print(len(retrieved_sentences))