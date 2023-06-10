import pickle
import sys

sys.path.append('../Parsing/')
from language_elements import Word, Chunk, Sentence

file_name = "../Parsing/full_dataset_113.pickle"

import os
print(os.getcwd())

with open(file_name, 'rb') as file:
    retrieved_sentences = pickle.load(file)

all_words = []
for sentence in retrieved_sentences:
    for chunk in sentence.list_of_chunks:
        for word in chunk.list_of_words:
            all_words.append(word.kannada_word)




def binary_search(arr, target):
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return True
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return False

summer = 0



with open('Embeddings/all_embeddings.pickle', 'rb') as file:
    all_embeddings = pickle.load(file)

    with open('Embeddings/embeddings_1_00_000.pickle', 'wb') as file:
        pickle.dump(all_embeddings[:1_00_000], file)
   
    # for embedding in all_embeddings:
    #     word = embedding[0]

    #     # if(binary_search(all_words, word)):
    #     #    summer+=1
    #     #    print(word)

    #     if(binary_search(all_words, word)):
    #        summer+=1
    #        print(word)
        

print(summer)
