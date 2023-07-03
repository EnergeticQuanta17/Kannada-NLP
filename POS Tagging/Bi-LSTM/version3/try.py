import pickle

with open('../../../Parsing/AnnotatedDatasetParsing/full_dataset_131.pickle', 'rb') as file:
    retrieved_sentences = pickle.load(file)

print("Number of sentences retrieved: ", len(retrieved_sentences))

words = []
tags = []

for sentence in retrieved_sentences:
    for chunk in sentence.list_of_chunks:
        for word in chunk.list_of_words:
            words.append(word.kannada_word)
            tags.append(word.pos)
    break

print(words, tags)