import pickle

with open('kan-index-to-char1.pkl', 'rb') as file:
    retrieved = pickle.load(file)
    
    sorted_dict = dict(sorted(retrieved.items(), key=lambda x: x[1]))

    for index, letter in sorted_dict.items():
        print(index, letter)
    
    print(len(retrieved))