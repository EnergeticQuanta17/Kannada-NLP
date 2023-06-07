import pickle

with open('kan-index-to-pos.pickle', 'rb') as file:
    retrieved = pickle.load(file)
    
    for index, letter in retrieved.items():
        print(index, letter)
    
    print(len(retrieved))