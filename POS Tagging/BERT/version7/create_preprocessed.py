import pickle
import os

from language_elements import Sentence, Chunk, Word

DATASET_PATH = "../../../Parsing/AnnotatedDatasetParsing/full_dataset_131.pickle"


with open(DATASET_PATH, 'rb') as file:
    retrieved_sentences = pickle.load(file)

with open('tokenizer_input.txt', 'w') as f:
    for sentence in retrieved_sentences:
        words = []
        for chunk in sentence.list_of_chunks:
            for word in chunk.list_of_words:
                words.append(word.kannada_word)
        f.write(" ".join(words) + '\n')


