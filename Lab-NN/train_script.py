import os
import sys
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser

from utils.train import train
from utils.scheduler import exponential_decay
from utils.data_processor import create_flower_dataloaders

from MLPAutoencoder import MLPAutoencoder, MLPAE_ENCODING_DIM
from VarAutoencoder import VarAutoencoder, VAE_loss_function, VAE_ENCODING_DIM
from Autoencoder import Autoencoder, AE_ENCODING_DIM

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="AE", choices=["VAE", "AE", "MLPAE"])
parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

# Set seeds
torch.manual_seed(0)
np.random.seed(0)


data_root = "./flowers"
model_save_root = "./model"

batch_size = args.batch_size
num_epochs = args.epochs
early_stopping_patience = 5
img_width, img_height = 24, 24
lr = args.lr

model = Autoencoder(AE_ENCODING_DIM) if args.model == 'AE' \
    else VarAutoencoder(VAE_ENCODING_DIM) if args.model == 'VAE' \
    else MLPAutoencoder(MLPAE_ENCODING_DIM, img_width, img_height)
loss_fn = VAE_loss_function if args.model == 'VAE' else F.mse_loss

if args.optimizer == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=lr)
else:
    optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = exponential_decay(
    initial_learning_rate=lr,
    decay_rate=0.9,
    decay_epochs=10,
)

training_dataloader, validation_dataloader = create_flower_dataloaders(batch_size, data_root, img_width, img_height)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
model.to(device)


# Start training
train(
    optimizer=optimizer,
    scheduler=scheduler,
    model=model,
    training_dataloader=training_dataloader,
    validation_dataloader=validation_dataloader,
    num_epochs=num_epochs,
    early_stopping_patience=early_stopping_patience,
    device=device,
    model_save_root=model_save_root,
    loss_fn=loss_fn,
)







