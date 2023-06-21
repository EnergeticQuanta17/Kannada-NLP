import os
from pickle import dump
from time import perf_counter

def extract_sentences(filename):
    pass

all_sentences = []

print("Starting to extract all sentences!")
start = perf_counter()
directory_path = r"..\..\Dataset\Raw Data"
for root, directories, files in os.walk(directory_path):
    for filename in files:
        file_path = os.path.join(root, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            index = 0
            for line in f:
                line = line.strip()
                words = line.split(' ')

                all_sentences.append(words)
                index += 1
print(f"Parsed all sentences. Time taken: {perf_counter()-start}")

with open('raw_sentences.pickle', 'wb') as file:
    dump(all_sentences, file)