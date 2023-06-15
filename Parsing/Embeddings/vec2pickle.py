import time
import pickle
import numpy as np

all_embeddings = []
embeddings_index = {}
words = []

start = time.time()
with open(r'..\..\..\Extracted-Kannada-Embeddings\cc.kn.300.vec', 'r', encoding='utf-8') as file:
    for index, line in enumerate(file):
        # print(line.strip())

        if(index==0):
            print(line)
            continue

        space_sep = line.strip().split(' ') 

        word = space_sep[0]
        float_list = [float(x) for x in space_sep[1:]]
        coefs = np.asarray(space_sep[1:], dtype='float32')        
        embeddings_index[word] = coefs
        
        embedding = (word, float_list)
        words.append(word)

        all_embeddings.append(embedding)

        if(index%200_000==0):
            print(time.time()-start)
            start = time.time()
            if(index==200_000):
                # with open('all_embeddings.pickle', 'wb') as file:
                #     pickle.dump(all_embeddings, file)  
                
                with open(f'embeddings_dict_{index//200000}.pickle', 'wb') as f:
                    pickle.dump(embeddings_index, f)

                with open('all_words.txt', 'w', encoding='utf-8') as f:
                    for word, embedding in all_embeddings:
                        f.write(word + '\n')
                
                all_embeddings = []
                words = []
                embeddings_index = {}
            
            elif(index>0):
                # with open('all_embeddings.pickle', 'ab') as file:
                #     pickle.dump(all_embeddings, file)

                with open(f'embeddings_dict_{index//200000}.pickle', 'wb') as f:
                    pickle.dump(embeddings_index, f)
                
                with open('all_words.txt', 'a', encoding='utf-8') as f:
                    for word, embedding in all_embeddings:
                        f.write(word + '\n')
                
                all_embeddings = []
                words = []
                embeddings_index = {}

print(len(all_embeddings))