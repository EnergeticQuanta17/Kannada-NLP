import pickle
from itertools import islice

with open('embeddings_dict.pickle', 'rb') as file:
    all_embeddings = pickle.load(file)

with open('embeddings_dict_10_000.pickle', 'wb') as file:
    embeddings_10_000 = dict(islice(all_embeddings.items(), 10000))
    pickle.dump(embeddings_10_000, file)