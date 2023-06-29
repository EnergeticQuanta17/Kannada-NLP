from tokenizers import ByteLevelBPETokenizer

input_files = ["kn_1k.txt"]

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=input_files, vocab_size=30000, min_frequency=2, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

tokenizer.save_model("kannada_tokenizer")

# print(dir(tokenizer))

sentence = "ನಮಸ್ತೆ ಕನ್ನಡ"

tokens = tokenizer.encode(sentence)

for token in tokens.tokens:
    print(bytes(token, 'utf-8'))
    print(bytes(token, 'utf-8').decode('unicode-escape'))
    print(bytes(token, 'utf-8').decode('unicode-escape').encode('latin-1'))
    print(bytes(token, 'utf-8').decode('unicode-escape').encode('latin-1').decode('utf-8'))
    
    break