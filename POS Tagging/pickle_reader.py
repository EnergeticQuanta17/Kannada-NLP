import pickle

with open('kan-index-to-char.pickle', 'r') as file:
    retrieved = pickle.load(file, encoding='utf-8')
    
    for index, letter in retrieved.items():
        print(index, letter)
    
    print(len(retrieved))