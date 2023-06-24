from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("l3cube-pune/kannada-bert")
text = "ನನಗೆ ಒಂದು ಪ್ರಶ್ನೆ ಇದೆ"

tokens = tokenizer.tokenize(text)
print(tokens)