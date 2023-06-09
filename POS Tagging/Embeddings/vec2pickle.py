import time
import pickle

all_embeddings = []
words = []

start = time.time()
with open(r'C:\Users\student\Downloads\cc.kn.300.vec\cc.kn.300.vec', 'r', encoding='utf-8') as file:
    for index, line in enumerate(file):
        # print(line.strip())

        if(index==0):
            print(line)
            continue

        space_sep = line.strip().split(' ')
        float_list = [float(x) for x in space_sep[1:]]
        
        embedding = (space_sep[0], float_list)
        words.append(space_sep[0])

        all_embeddings.append(embedding)

        if(index%200_000==0):
            print(index, time.time()-start)
            start = time.time()
            if(index==200_000):
                all_embeddings.sort()
                with open('all_embeddings.pickle', 'wb') as file:
                    pickle.dump(all_embeddings, file)  

                with open('all_words.txt', 'w', encoding='utf-8') as f:
                    for word, embedding in all_embeddings:
                        f.write(word + '\n')
                
                all_embeddings = []
                words = []
            
            elif(index>0):
                all_embeddings.sort()
                with open('all_embeddings.pickle', 'ab') as file:
                    pickle.dump(all_embeddings, file)
                
                with open('all_words.txt', 'a', encoding='utf-8') as f:
                    for word, embedding in all_embeddings:
                        f.write(word + '\n')
                
                all_embeddings = []
                words = []

print(len(all_embeddings))