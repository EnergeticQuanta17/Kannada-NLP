from tokenizers import ByteLevelBPETokenizer

input_files = ["kn_1k.txt"]

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=input_files, vocab_size=30000, min_frequency=2, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

tokenizer.save_model("kannada_tokenizer")

# print(dir(tokenizer))

sentence = "ನಮಸ್ತೆ ಕನ್ನಡ"
print(sentence, sentence.encode('utf-8').decode('utf-8'))
tokens = tokenizer.encode(sentence)
print(type(tokens.tokens))

all_tokens = " ".join(tokens.tokens)
print(all_tokens, all_tokens.encode('utf-8'), all_tokens.decode('utf-8'))

# for i in tokens.tokens:
#     print(i.encode('utf-8').decode('unicode_escape'))