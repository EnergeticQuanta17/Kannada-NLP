import pickle

import torch
import torch.nn as nn
import torch.optim as optim

EPOCHS = 10

class BiLSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(BiLSTMTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, words):
        print(words)
        words = torch.tensor(words)
        embeds = self.embedding(words)
        lstm_out, _ = self.lstm(embeds)
        logits = self.hidden2tag(lstm_out)
        return logits

def train(model, train_data, optimizer, criterion):
    for epoch in range(EPOCHS):
        for words, tags in train_data:
            logits = model(words)
            loss = criterion(logits, tags)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def evaluate(model, test_data):
    correct = 0
    total = 0
    for words, tags in test_data:
        logits = model(words)
        _, predicted = torch.max(logits, 1)
        total += len(tags)
        correct += (predicted == tags).sum().item()
    return correct / total

if __name__ == "__main__":
    embedding_dim = 100
    hidden_dim = 100
    vocab_size = 15000
    tagset_size = 78

    model = BiLSTMTagger(embedding_dim, hidden_dim, vocab_size, tagset_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    with open('../../../Parsing/AnnotatedDatasetParsing/full_dataset_131.pickle', 'rb') as file:
        retrieved_sentences = pickle.load(file)
    
    print(len(retrieved_sentences))
    
    train_data = []
    for sentence in retrieved_sentences[:5686]:
        temp_words = []
        temp_tags = []
        for chunk in sentence.list_of_chunks:
            for word in chunk.list_of_words:
                temp_words.append(word.kannada_word)
                temp_tags.append(word.pos)
        train_data.append((temp_words, temp_tags))
        
    test_data = []
    for sentence in retrieved_sentences[5686:]:
        temp_words = []
        temp_tags = []
        for chunk in sentence.list_of_chunks:
            for word in chunk.list_of_words:
                temp_words.append(word.kannada_word)
                temp_tags.append(word.pos)
        test_data.append((temp_words, temp_tags))


    train(model, train_data, optimizer, criterion)
    print("Accuracy on test set:", evaluate(model, test_data))
