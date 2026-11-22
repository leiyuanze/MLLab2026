import torch
from torch import nn
from torch.nn import functional as F

VAE_ENCODING_DIM = 64

# Define the Variational Encoder
class VarEncoder(nn.Module):
    def __init__(self, encoding_dim, image_size=24):
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        super(VarEncoder, self).__init__()
        # TODO: implement the encoder
        if image_size % 4 != 0:
            raise ValueError("image_size must be divisible by 4")
        self.spatial_size = image_size // 4
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        feature_count = 128 * self.spatial_size * self.spatial_size
        self.fc_mu = nn.Linear(feature_count, encoding_dim)
        self.fc_log_var = nn.Linear(feature_count, encoding_dim)


    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return mu: mean of the distribution, dim: (Batch_size, encoding_dim)
        return log_var: log of the variance of the distribution, dim: (Batch_size, encoding_dim)
        '''
        
        # TODO: implement the forward pass
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

# Define the Decoder
class VarDecoder(nn.Module):
    def __init__(self, encoding_dim, image_size=24):
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        super(VarDecoder, self).__init__()
        # TODO: implement the decoder
        if image_size % 4 != 0:
            raise ValueError("image_size must be divisible by 4")
        self.spatial_size = image_size // 4
        self.fc = nn.Linear(
            encoding_dim,
            128 * self.spatial_size * self.spatial_size,
        )
        self.features = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, v):
        '''
        v: latent vector, dim: (Batch_size, encoding_dim)
        return x: reconstructed images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        '''
        
        # TODO: implement the forward pass
        x = self.fc(v).reshape(
            v.shape[0],
            128,
            self.spatial_size,
            self.spatial_size,
        )
        x = self.features(x)
        return x

# Define the Variational Autoencoder
class VarAutoencoder(nn.Module):
    def __init__(self, encoding_dim, image_size=24):
        super(VarAutoencoder, self).__init__()
        self.encoder = VarEncoder(encoding_dim, image_size)
        self.decoder = VarDecoder(encoding_dim, image_size)

    @property
    def name(self):
        return "VAE"

    def reparameterize(self, mu, log_var):
        '''
        mu: mean of the distribution, dim: (Batch_size, encoding_dim)
        log_var: log of the variance of the distribution, dim: (Batch_size, encoding_dim)
        return v: sampled latent vector, dim: (Batch_size, encoding_dim)
        '''
        
        
        # TODO: implement the reparameterization trick to sample v
        epsilon = torch.randn_like(log_var)
        z = mu + torch.exp(0.5 * log_var) * epsilon
        return z
        
    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return x: reconstructed images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return mu: mean of the distribution, dim: (Batch_size, encoding_dim)
        return log_var: log of the variance of the distribution, dim: (Batch_size, encoding_dim)
        '''
        # TODO: implement the forward pass
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x = self.decoder(z)
        return x, mu, log_var

# Loss Function
def VAE_loss_function(outputs, images):
    '''
    outputs: (x, mu, log_var)
    images: input/original images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
    return loss: the loss value, dim: (1)
    '''
    # TODO: implement the loss function for VAE
    reconstruction, mu, log_var = outputs
    reconstruction_loss = F.mse_loss(reconstruction, images, reduction="sum")
    kl_divergence = -0.5 * torch.sum(
        1 + log_var - mu.pow(2) - log_var.exp()
    )
    return reconstruction_loss + kl_divergence
