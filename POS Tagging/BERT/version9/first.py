from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("l3cube-pune/kannada-bert")
text = "ನನಗೆ ಒಂದು ಪ್ರಶ್ನೆ ಇದೆ"
tokens = tokenizer.tokenize(text)

input_ids = tokenizer.convert_tokens_to_ids(tokens)

import torch

input_ids = torch.tensor([input_ids])
attention_mask = torch.ones_like(input_ids)
token_type_ids = torch.zeros_like(input_ids)

# Move tensors to the appropriate device if using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
token_type_ids = token_type_ids.to(device)


from transformers import BertModel

model = BertModel.from_pretrained("l3cube-pune/kannada-bert")
model.to(device)

outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)


