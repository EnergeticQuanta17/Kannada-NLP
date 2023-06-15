from transformers import BertModel, BertConfig
import torch.nn as nn
import torch

# Load the pre-trained BERT model
model = BertModel.from_pretrained('bert-base-cased')

# Get the configuration of the pre-trained model
config = model.config

# Modify the configuration to change the embedding size
config.hidden_size = 1024  # Set the desired embedding size

# Create a new embedding layer with the modified size
new_embedding_layer = model.embeddings.word_embeddings

# Modify the size of the new embedding layer
new_embedding_layer = new_embedding_layer.requires_grad_(False)  # Freeze the new embeddings
new_embedding_layer.weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(config.vocab_size, config.hidden_size)))

# Update the model with the new embedding layer
model.embeddings.word_embeddings = new_embedding_layer

# Print the modified model
print(model)
