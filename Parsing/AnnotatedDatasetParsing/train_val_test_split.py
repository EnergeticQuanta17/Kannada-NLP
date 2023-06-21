import csv
import random

def split_dataset(dataset, slice_ratios):
    dataset_size = len(dataset)
    train_size = int(dataset_size * slice_ratios[0])
    val_size = int(dataset_size * slice_ratios[1])
    test_size = dataset_size - train_size - val_size

    start_indices = random.sample(range(0, dataset_size - (val_size+test_size) + 1), 2)
    start_indices.sort()

    train_set = []
    val_set = dataset[start_indices[1]:start_indices[1]+val_size]
    test_set = dataset[start_indices[1]+val_size:start_indices[1]+val_size+test_size]

    not_train_indices = list(range(start_indices[1], start_indices[1]+val_size)) + list(range(start_indices[1]+val_size, start_indices[1]+val_size+test_size))

    train_indices = list(set(range(dataset_size)) - set(not_train_indices))
    train_set = [dataset[i] for i in train_indices]

    return train_set, val_set, test_set

# dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# slice_ratios = [0.8, 0.1, 0.1]

# train_data, val_data, test_data = split_dataset(dataset, slice_ratios)

# print("Training data:", train_data)
# print("Validation data:", val_data)
# print("Test data:", test_data)

##########################################################################

import pickle

from language_elements import Sentence, Word, Chunk

file_name = input('FileName: ')

full_dataset_sentences = []
with open(file_name, 'rb') as file:
    retrieved_sentences = pickle.load(file)

train_data, val_data, test_data = split_dataset(retrieved_sentences, [0.8, 0.1, 0.1])
print(type(train_data[0]))

def x_y(data):
    x = []
    y = []
    for sentence in data:
        for chunk in sentence.list_of_chunks:
            for word in chunk.list_of_words:
                x.append(word.kannada_word)
                y.append(word.pos)
    
    return zip(x, y)

train_data = x_y(train_data)
val_data = x_y(val_data)
test_data = x_y(test_data)

def write_to_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Kannada-Word', 'Part-of-Speech'])  # Write header row
        writer.writerows(data)

write_to_csv(train_data, "train.csv")
write_to_csv(val_data, "validation.csv")
write_to_csv(test_data, "test.csv")