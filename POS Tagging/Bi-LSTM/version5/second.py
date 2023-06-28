import torch

def embedding(weight, indices):
    return torch.embedding(weight, tensor_indices)

if __name__ == "__main__":
    weight = torch.randn(1000, 100)
    vocab = {"The": 1, "quick": 2, "brown": 3, "fox": 4, "jumps": 5, "over": 6, "the": 7, "lazy": 8, "dog": 9}
    words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    indices = [vocab[word] for word in words]
    tensor_indices = torch.tensor(indices)
    embeds = embedding(weight, tensor_indices)
    print(embeds)
