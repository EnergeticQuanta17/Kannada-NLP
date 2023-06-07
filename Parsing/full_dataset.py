import os
import pickle

from dataset_parser import extract_list_of_sentences

file_names = os.listdir(r'.\Dataset\DL-DL MT\DL-DL MT')

# for i in file_names:
#     print(i)

print(len(file_names))

all_sentences = []
sum =0
for file in file_names:
    sentences = extract_list_of_sentences(file)
    print(file, len(sentences))
    sum += len(sentences)
    all_sentences += sentences
print(len(all_sentences), sum)
with open('full_dataset_113.pickle', 'wb') as file:
    pickle.dump(all_sentences, file)