import pickle
from itertools import islice

with open('embeddings_dict.pickle', 'rb') as file:
    all_embeddings = pickle.load(file)

print(len(all_embeddings))

# with open('embeddings_dict_1_000_000.pickle', 'wb') as file:
#     embeddings_1_000_000 = dict(islice(all_embeddings.items(), 1000000))
#     pickle.dump(embeddings_1_000_000, file)