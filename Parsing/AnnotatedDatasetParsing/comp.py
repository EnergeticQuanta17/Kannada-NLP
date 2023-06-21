import pickle
import time

from language_elements import Sentence, Word, Chunk

file_name = "full_dataset_131.pickle"

all_words = set()

with open(file_name, 'rb') as file:
    retrieved_sentences = pickle.load(file)
    for sentence in retrieved_sentences:
        for chunk in sentence.list_of_chunks:
            for word in chunk.list_of_words:
                all_words.add(word.kannada_word)
        

all_words_dataset = list(all_words)
print("Number of words in the dataset: ", len(all_words_dataset))

##########################################################

for iiiii in range(3, 9):
    file_name = f"Embeddings/embeddings_dict_{iiiii}.pickle"
    all_words = set()
    with open(file_name, 'rb') as file:
        emb_dict = pickle.load(file)
        for kan_word in emb_dict:
            all_words.add(kan_word)

    all_words_embeddings = list(all_words)
    print("Number of words in the Embeddings: ", len(all_words_embeddings))


    print(all_words_embeddings[0].encode('utf-8'))

    # count = 0
    # for word_dataset in all_words_dataset:
    #     for word_emb in all_words_embeddings:
    #         if(word_dataset.encode('utf-8')==word_emb.encode('utf-8')):
    #             # print(word_dataset, word_emb)
    #             count += 1
    # print(f"Count of matching words between {len(all_words_embeddings)} embeddings and words: {count}")

    all_words_dataset_utf8 = sorted([i.encode('utf-8') for i in all_words_dataset])
    all_words_embeddings_utf8 = sorted([i.encode('utf-8') for i in all_words_embeddings])

    print(f'\nStarting to count for {iiiii}')
    start = time.perf_counter()
    count = 0
    for word_dataset in all_words_dataset_utf8:
        if(word_dataset in all_words_embeddings_utf8):
            count+=1
    print(count, time.perf_counter()-start)
