from tokenizers import ByteLevelBPETokenizer
import codecs
import chardet

input_files = ["kn.txt"]

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=input_files, vocab_size=30000, min_frequency=2, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

tokenizer.save_model("kannada_tokenizer")

# print(dir(tokenizer))

sentence = "ನಮಸ್ತೆ ಕನ್ನಡ"

tokens = tokenizer.encode(sentence)

for token in tokens.tokens:
    print(token)