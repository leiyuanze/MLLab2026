import torch
import torch.nn as nn
import torch.nn.functional as F

AE_ENCODING_DIM = 64

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, encoding_dim, image_size=24):
        super(Encoder, self).__init__()
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        
        '''
        TODO: Implement the Encoder.

        Requirements:
        1. Use convolutional layers to extract features from the input images.
        2. Apply max pooling to downsample the spatial dimensions.
        3. Use a linear layer to map the feature maps to the latent vector.
        '''
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
        self.fc = nn.Linear(
            128 * self.spatial_size * self.spatial_size,
            encoding_dim,
        )

    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return v: latent vector, dim: (Batch_size, encoding_dim)
        '''
        
        '''
        TODO: Implement the forward pass of the Encoder.

        Steps:
        1. Pass the input through the convolutional layers and max pooling.
        2. Flatten the output and pass it through the linear layer to obtain the latent vector.
        3. Return the latent vector.
        '''
        x = self.features(x)
        v = self.fc(torch.flatten(x, start_dim=1))
        return v


# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, encoding_dim, image_size=24):
        super(Decoder, self).__init__()
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        
        '''
        TODO: Implement the Decoder.

        Requirements:
        1. Use a linear layer to map the latent vector back to the feature map dimensions.
        2. Use transposed convolutional layers to upsample the feature maps.
        3. Ensure the output has the same dimensions as the input image.
        '''
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
        
        '''
        TODO: Implement the forward pass of the Decoder.

        Steps:
        1. Pass the latent vector through the linear layer to reconstruct the feature maps.
        2. Pass the feature maps through transposed convolutional layers to upsample them.
        3. Return the reconstructed images.
        '''
        x = self.fc(v).reshape(
            v.shape[0],
            128,
            self.spatial_size,
            self.spatial_size,
        )
        x = self.features(x)
        return x


# Combine the Encoder and Decoder to make the autoencoder
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim, image_size=24):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(encoding_dim, image_size)
        self.decoder = Decoder(encoding_dim, image_size)

    def forward(self, x):
        '''
        TODO: Implement the forward pass of the Autoencoder.

        Steps:
        1. Pass the input through the Encoder to obtain the latent vector.
        2. Pass the latent vector through the Decoder to reconstruct the input.
        3. Return the reconstructed images.
        '''
        v = self.encoder(x)
        x = self.decoder(v)
        return x
    
    @property
    def name(self):
        return "AE"
