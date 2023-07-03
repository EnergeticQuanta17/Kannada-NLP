import pickle
import sys

with open('../../../Parsing/AnnotatedDatasetParsing/full_dataset_131.pickle', 'rb') as file:
    retrieved_sentences = pickle.load(file)

print("Number of sentences retrieved: ", len(retrieved_sentences))

# SENTENCE_INDEX = int(input("Enter sentence index: "))
SENTENCE_INDEX = int(sys.argv[1])
print(sys.argv[1])
raise Exception

words = []
tags = []

sentence = retrieved_sentences[SENTENCE_INDEX]
for chunk in sentence.list_of_chunks:
    for word in chunk.list_of_words:
        words.append(word.kannada_word)
        tags.append(word.pos)

print(words, tags)

with open('input_words.txt', 'w') as f:
    for word in words:
        f.write(word + '\n')


with open('input_tags.txt', 'w') as f:
    for tag in tags:
        f.write(tag + '\n')
