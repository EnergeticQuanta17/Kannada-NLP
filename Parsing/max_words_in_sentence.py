import pickle

from language_elements import Sentence, Word, Chunk

file_name = "full_dataset_113.pickle"

full_dataset_sentences = []

max_words = 0

with open(file_name, 'rb') as file:
    retrieved_sentences = pickle.load(file)
    for sentence in retrieved_sentences:
        count = 0
        for chunk in sentence.list_of_chunks:
            for word in chunk.list_of_words:
                count += 1
        max_words = max(max_words, count)

print(max_words)