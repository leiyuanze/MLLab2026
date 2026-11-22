import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from VarAutoencoder import VarAutoencoder, VAE_ENCODING_DIM
from Autoencoder import Autoencoder, AE_ENCODING_DIM


# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument("--model", type=str, default="VAE", choices=["VAE", "AE"])
args = parser.parse_args()

# Set paths and model configuration based on the selected model
model_save_path = f"./model/Best_{args.model}.pth"
vis_root = "./vis"
os.makedirs(vis_root, exist_ok=True)

model_class = VarAutoencoder if args.model == "VAE" else Autoencoder
ENCODING_DIM = AE_ENCODING_DIM if args.model == "AE" else VAE_ENCODING_DIM

# Initialize the selected model and load its parameters
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
model = model_class(encoding_dim=ENCODING_DIM)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)

# Ensure the model is in evaluation mode
model.eval()

# TODO: Generate random images
'''
Steps:
1. Sample 10 latent vectors from a standard normal distribution (mean=0, std=1).
2. Pass the sampled latent vectors through the decoder to generate images.
3. Ensure the generated images are stored in a tensor `random_images` with shape (10, 3, 24, 24) and are in valid range [0, 1].
'''
with torch.no_grad():
    latent_vectors = torch.randn(10, ENCODING_DIM, device=device)
    random_images = model.decoder(latent_vectors).clamp(0.0, 1.0).cpu()

# Save the 10 random images in one figure
fig = plt.figure(figsize=(10, 1))

for i in range(10):
    ax = fig.add_subplot(1, 10, i + 1, xticks=[], yticks=[])
    # Convert tensor to numpy array and transpose dimensions for visualization
    ax.imshow(np.transpose(random_images[i].detach().numpy(), (1, 2, 0)))

# Save the generated images
os.makedirs(vis_root, exist_ok=True)
plt.savefig(f"{vis_root}/random_images_{args.model}.png")
plt.close(fig)
