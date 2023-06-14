import os
import json

BERT_MODEL_NAMES = [
    'bert-base-uncased',
    'bert-large-uncased',
    'bert-base-multilingual-cased',
    'bert-large-uncased-whole-word-masking',  
]

print("Availabel BERT Models:", BERT_MODEL_NAMES)

if(os.path.exists('configuration.json')):
    with open('configuration.json', 'r') as json_file:
        config = json.load(json_file)
else:
    config = {
        'batch_size': 16,
        'epochs': 1,
        'epochs_stagnate':10,
        'bert-model-name':"bert-base-uncased",
    }

DATASET = [
    '../../../Parsing/full_dataset_113.pickle',
    '../../../Parsing/full_dataset_131.pickle'
]

types = ['int', 'int', 'int', 'int']

for index, item in enumerate(config.items()):
    key, value = item

    if(key=='bert-model-name'):
        print("\n", BERT_MODEL_NAMES)

    print(f"{key}: {value}. Change? ", end='')

    while(True):
        inp = input()

        if(inp==''):
            break
        try:
            int(inp)
            config[key] = int(inp)
            break
        except:
            print("Change? ")
            pass


with open('configuration.json', 'w') as json_file:
    json.dump(config, json_file)