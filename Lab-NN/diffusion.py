import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from utils.scheduler import exponential_decay
from utils.data_processor import create_flower_dataloaders


# Set random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Basic settings
data_root = "./flowers"  # Path to the dataset root directory
model_save_path = "./model/BestModel_diffusion.pkl"  # Path to save the trained model
vis_root = "./vis"

# Hyperparameters (adjustable)
batch_size = 16  # Batch size for training and validation
num_epochs = 1000  # Number of training epochs
img_channel = 3
img_width, img_height = 24, 24

device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch, "mps") and torch.mps.is_available() else "cpu"
print(f"using device {device}")

# Diffusion process settings (adjustable)
num_steps = 300  # Number of steps in the diffusion process
beta_min = 1e-4  # Minimum value of the beta schedule
beta_max = 5e-2  # Maximum value of the beta schedule

# Initialize diffusion parameters
# These parameters define the noise schedule and are crucial for the forward/reverse process
betas = torch.linspace(-6, 6, num_steps, device=device) # Linearly spaced values for noise schedule
betas = torch.sigmoid(betas) * (beta_max - beta_min) + beta_min  # Scale sigmoid outputs
alphas = 1 - betas  # Compute alpha values from beta
alphas_sqrt = torch.sqrt(alphas)  # Square root of alphas
alphas_bar = torch.cumprod(alphas, 0)  # Cumulative product of alphas over steps
alphas_bar_sqrt = torch.sqrt(alphas_bar)  # Square root of cumulative alphas
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_bar)  # Square root of 1 - cumulative alphas


# Define the diffusion model
class Diffuser(nn.Module):
    def __init__(self, n_steps, img_channels=3):
        """
        Initialize a diffusion model. The model predicts the added noise at each time step.

        Args:
          - n_steps (int): Number of diffusion steps (t).
          - img_channels (int): Number of image channels (e.g., 3 for RGB images).
        """
        super(Diffuser, self).__init__()
        # TODO: Define an encoder-decoder architecture for epsilon 
        # prediction, and add time step embeddings to condition the 
        # predictions.
        self.n_steps = n_steps
        time_dim = 128
        self.time_embedding = nn.Embedding(n_steps, time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.mid = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.up1 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.up2 = nn.ConvTranspose2d(
            64, img_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x, t):
        """
        Forward pass of the diffusion model:
        Predict the noise added to x at a given time step t.

        Input:
          - x (torch.Tensor): Noisy input image, shape (batch_size, img_channels, height, width).
          - t (torch.Tensor): Time step indices, shape (batch_size,).
        Output:
          - Predicted noise, shape same as x.
        """
        # TODO: Implement the forward computation for a diffusion model
        t = t.long()
        time_embedding = self.time_mlp(self.time_embedding(t))
        time_embedding = time_embedding[:, :, None, None]

        h1 = F.silu(self.enc1(x))
        h2 = F.silu(self.enc2(h1))
        h2 = self.mid(h2 + time_embedding)
        h = F.silu(self.up1(h2)) + h1
        h = self.block1(h)
        return self.up2(h)

    def sample(self, shape, n_steps):
        """
        Generate images by iteratively sampling from pure noise.

        Args:
        - shape (tuple): Shape of the generated images (batch_size, img_channels, height, width).
        - n_steps (int): Number of diffusion steps.

        Returns:
        - x_seq (list): List of images at each step of the reverse process.
        """
        self.eval()

        x_t = torch.randn(shape, device=next(self.parameters()).device)
        x_seq = [x_t]
        for t in reversed(range(n_steps)):
            x_t = self.p_theta_sampling(x_t, t)
            x_seq.append(x_t)
        return x_seq

    def p_theta_sampling(self, x, t):
        """
        Estimate x[t-1] given x[t] using the reverse diffusion process.

        Steps:
        1. Predict the noise added to x[t] using the model.
        2. Compute the mean of the reverse process based on the formula.
        3. Optionally add Gaussian noise for stochasticity if t > 0.

        Args:
        - model (Diffuser): The diffusion model.
        - x (torch.Tensor): Image at time step t, shape (batch_size, img_channels, height, width).
        - t (int): Current time step.

        Returns:
        - x_t_minus_1 (torch.Tensor): Estimated image at time step t-1.
        """
        # TODO:
        # Predict noise using the model and compute the mean for the reverse process.
        # Add Gaussian noise if t > 0 to simulate the stochastic nature of the process.
        batch_size = x.shape[0]
        t_tensor = torch.full(
            (batch_size,), t, dtype=torch.long, device=x.device
        )
        predicted_noise = self(x, t_tensor)

        beta_t = betas.to(x.device)[t].reshape(1, 1, 1, 1)
        alpha_t = alphas.to(x.device)[t].reshape(1, 1, 1, 1)
        alpha_bar_t = alphas_bar.to(x.device)[t].reshape(1, 1, 1, 1)
        if t > 0:
            alpha_bar_previous = alphas_bar.to(x.device)[t - 1].reshape(
                1, 1, 1, 1
            )
        else:
            alpha_bar_previous = torch.ones_like(alpha_bar_t)

        mean = (
            beta_t * torch.sqrt(alpha_bar_previous) / (1 - alpha_bar_t)
        ) * x
        mean += (
            (1 - alpha_bar_previous)
            * torch.sqrt(alpha_t)
            / (1 - alpha_bar_t)
        ) * predicted_noise

        if t > 0:
            posterior_variance = (
                beta_t * (1 - alpha_bar_previous) / (1 - alpha_bar_t)
            )
            noise = torch.randn_like(x)
            x_t_minus_1 = mean + torch.sqrt(posterior_variance) * noise
        else:
            x_t_minus_1 = mean
        return x_t_minus_1

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

# Loss function for training
def diffusion_loss_fn(model, x_0):
    """
    Compute the training loss for the diffusion model:
    This function compares the model's predicted noise with the actual noise.

    Steps:
      1. Randomly sample a batch of time steps t.
      2. Add Gaussian noise to x_0 to simulate the forward process.
      3. Predict the noise using the model.
      4. Compute the mean squared error between predicted and actual noise.

    Args:
      - model (Diffuser): The diffusion model.
      - x_0 (torch.Tensor): Original clean image, shape (batch_size, img_channels, height, width).

    Returns:
      - loss (torch.Tensor): Scalar value representing the loss.
    """
    batch_size = x_0.size(0)
    t = torch.randint(0, num_steps, size=(batch_size,), device=x_0.device)
    # Randomly sample time steps
    # TODO:
    # Compute coefficients for x_0 and noise at step t.
    # Generate noisy images using the diffusion forward process.
    # Use the model to predict noise and compute MSE loss.
    noise = torch.randn_like(x_0)
    alpha_bar_t = alphas_bar.to(x_0.device)[t].reshape(-1, 1, 1, 1)
    noisy_images = (
        torch.sqrt(alpha_bar_t) * x_0
        + torch.sqrt(1 - alpha_bar_t) * noise
    )
    predicted_noise = model(noisy_images, t)
    return torch.mean((predicted_noise - noise) ** 2)

# Sampling function with visualization
def visualize_sampling(model, sample_shape, n_steps, save_path):
    """Visualize the sampling process during training."""
    if not os.path.splitext(save_path)[1]:
        save_path += ".png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()
    with torch.no_grad():
        x_seq = model.sample(sample_shape, n_steps)
    
    num_shows = min(10, n_steps)  # Number of steps to display
    snapshot_indices = np.linspace(
        0, len(x_seq) - 1, num=num_shows, dtype=int
    )
    remaining_steps = np.linspace(
        n_steps, 0, num=num_shows, dtype=int
    )
    fig, axes = plt.subplots(1, num_shows, figsize=(15, 5))
    for i, ax in enumerate(axes):
        sequence_index = snapshot_indices[i]
        image = (
            x_seq[sequence_index]
            .squeeze()
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        
        # Normalize the image to [-1, 1]
        image_min, image_max = image.min(), image.max()
        normalized_image = (
            2 * (image - image_min) / max(image_max - image_min, 1e-8) - 1
        )
        
        # Clip to [-1, 1] for visualization (optional)
        normalized_image = normalized_image.clip(-1, 1)
        
        # Shift back to [0, 1] for imshow visualization
        imshow_image = (normalized_image + 1) / 2  # [0, 1]
        
        ax.imshow(imshow_image)
        ax.axis('off')
        ax.set_title(f"t={remaining_steps[i]}")
    plt.suptitle(f"Sampling Visualization")
    plt.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=num_epochs)
    args = parser.parse_args()
    num_epochs = args.epochs

    # Load data
    training_dataloader, validation_dataloader = create_flower_dataloaders(
        batch_size, data_root, img_width, img_height
    )

    # Initialize model, optimizer, and scheduler
    model = Diffuser(n_steps=num_steps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = exponential_decay(initial_learning_rate=1e-3, decay_rate=0.9, decay_epochs=5)

    sample_shape = (1, img_channel, img_width, img_height)  # Shape for visualization samples

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        learning_rate = scheduler(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate
        train_losses = []
        for images, _ in training_dataloader:
            images = images.to(device).float()
            # normalize images input to [-1, 1]
            images = 2 * images - 1
            loss = diffusion_loss_fn(model, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = sum(train_losses) / len(train_losses)

        print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # Visualize sampling process every 10 epochs
        if (epoch + 1) % 10 == 0:
            visualize_sampling(model, sample_shape, num_steps, os.path.join(vis_root, f"random_images_diffusion_epoch_{epoch + 1}"))

    model.save(model_save_path)
    model.save("./model/Best_diffusion.pth")
    model.save("./model/BestModel_diffusion.pkl")
    visualize_sampling(model, sample_shape, num_steps, os.path.join(vis_root, "random_images_diffusion"))
