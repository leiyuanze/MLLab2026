import torch
import torch.nn as nn
import torch.nn.functional as F

MLPAE_ENCODING_DIM = 64

class MLPAutoencoder(nn.Module):
    def __init__(self, encoding_dim, img_width, img_height, img_channel=3):
        super(MLPAutoencoder, self).__init__()
        '''
        TODO: Define the MLP autoencoder structure.
        
        Steps:
        1. Design the encoder with multiple layers to reduce the dimensionality.
        2. Design the decoder to reconstruct the input from the encoded representation.
        3. Ensure the architecture supports end-to-end training.
        '''
        input_dim = img_channel * img_width * img_height
        self.input_shape = (img_channel, img_width, img_height)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, IMG_CHANNEL, IMG_WIDTH, IMG_HEIGHT)
        return x: reconstructed images, dim: (Batch_size, IMG_CHANNEL, IMG_WIDTH, IMG_HEIGHT)

        TODO: Define the forward pass of the model.
        
        Steps:
        1. Process the input through the encoder.
        2. Pass the encoded representation through the decoder.
        3. Return the reconstructed output.
        '''
        batch_size = x.shape[0]
        v = self.encoder(torch.flatten(x, start_dim=1))
        x = self.decoder(v)
        return x.reshape(batch_size, *self.input_shape)
    
    @property
    def name(self):
        return "MLPAE"
