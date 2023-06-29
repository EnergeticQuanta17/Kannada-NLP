from transformers import BertWordPieceTokenizer

input_files = ["kn_1k.txt"]

tokenizer = BertWordPieceTokenizer()
tokenizer.train(files=input_files, vocab_size=30000, min_frequency=2, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], only_alphabet=True)

tokenizer.save_model("kannada_tokenizer")

sentence = "ನಮಸ್ತೆ ಕನ್ನಡ"
tokens = tokenizer.tokenize(sentence)
print(tokens)