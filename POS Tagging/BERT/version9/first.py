from transformers import BertModel

model_name = "l3cube-pune/kannada-bert"
model = BertModel.from_pretrained(model_name)

print(model)