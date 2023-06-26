import torch

# Create a 2D tensor
tensor = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], requires_grad=True)

# Create a mask to specify which elements should not require gradients
mask = torch.tensor([0, 1], dtype=torch.bool)

# Disable gradients for masked elements
masked_tensor = tensor[:, mask].detach()

# Print the tensors and requires_grad status
print(tensor)
print(tensor.requires_grad)
print(masked_tensor)
print(masked_tensor.requires_grad)