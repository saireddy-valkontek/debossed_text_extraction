import torch

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
state_dict = torch.load("crnn_model.pth", map_location=device)

# Print model keys and shapes
for key in state_dict.keys():
    print(key, state_dict[key].shape)
