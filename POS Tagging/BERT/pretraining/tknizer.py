from tokenizers import ByteLevelBPETokenizer

input_files = ["kn_1k.txt"]

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=input_files, vocab_size=30000, min_frequency=2, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

tokenizer.save_model("kannada_tokenizer")

# print(dir(tokenizer))

sentence = "ನಮಸ್ತೆ ಕನ್ನಡ"
# sentence = "ಡ"
# print(chr(ord(sentence)))
print(sentence, sentence.encode('utf-8').decode('utf-8'))
tokens = tokenizer.encode(sentence)
print(type(tokens.tokens))

for token in tokens.tokens:
    print(token)