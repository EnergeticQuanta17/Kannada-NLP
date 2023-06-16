from datasets import load_dataset
dataset = load_dataset("Energetic26/kannada_pos")

print(type(dataset))
print(dataset)

from tqdm.auto import tqdm

text_data = []
file_count = 0

for sample in tqdm(dataset['train']):
    sample = sample['Kannada-Word'].replace('\n', '')
    text_data.append(sample)
    if len(text_data) == 10_000:
        with open(f'data/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))
        text_data = []
        file_count += 1

with open(f'data/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
    fp.write('\n'.join(text_data))

################################################################################

from pathlib import Path
from tokenizers import BertWordPieceTokenizer

paths = [str(x) for x in Path('data').glob('**/*.txt')]

tokenizer = BertWordPieceTokenizer()

tokenizer.train(files=paths, vocab_size=10000000, min_frequency=2,
                special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

################################################################################

import os
os.mkdir('./kanbert')
tokenizer.save_model('kanbert')