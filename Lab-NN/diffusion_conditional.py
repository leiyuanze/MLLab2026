import json
import math
import os
import time
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets

from utils.data_processor import create_flower_dataloaders


def parse_args():
    parser = ArgumentParser(
        description="Train a class-conditional DDPM on the flower dataset."
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--label-dropout", type=float, default=0.1)
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "mps") and torch.mps.is_available():
        return "mps"
    return "cpu"


class ConditionalResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, condition_dim):
        super().__init__()
        groups = min(8, out_channels)
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.condition = nn.Linear(condition_dim, out_channels * 2)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, condition):
        hidden = F.silu(self.norm1(x))
        hidden = self.conv1(hidden)
        scale, shift = self.condition(condition).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        hidden = hidden * (1 + scale) + shift
        hidden = F.silu(self.norm2(hidden))
        hidden = self.conv2(hidden)
        return (hidden + self.shortcut(x)) / math.sqrt(2)


class ConditionalDiffuser(nn.Module):
    def __init__(
        self,
        num_steps,
        num_classes,
        img_channels=3,
        base_channels=64,
        condition_dim=128,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.null_class = num_classes
        self.condition_dim = condition_dim

        beta_min, beta_max = 1e-4, 5e-2
        schedule = torch.linspace(-6, 6, num_steps)
        betas = torch.sigmoid(schedule) * (beta_max - beta_min) + beta_min
        alphas = 1 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)

        self.time_embedding = nn.Embedding(num_steps, condition_dim)
        self.class_embedding = nn.Embedding(
            num_classes + 1, condition_dim
        )
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

        self.input_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)
        self.down1 = nn.Conv2d(
            base_channels, base_channels * 2, 4, stride=2, padding=1
        )
        self.res1 = ConditionalResBlock(
            base_channels * 2, base_channels * 2, condition_dim
        )
        self.down2 = nn.Conv2d(
            base_channels * 2, base_channels * 4, 4, stride=2, padding=1
        )
        self.res2 = ConditionalResBlock(
            base_channels * 4, base_channels * 4, condition_dim
        )
        self.mid1 = ConditionalResBlock(
            base_channels * 4, base_channels * 4, condition_dim
        )
        self.mid2 = ConditionalResBlock(
            base_channels * 4, base_channels * 4, condition_dim
        )
        self.up1 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            4,
            stride=2,
            padding=1,
        )
        self.up_block1 = ConditionalResBlock(
            base_channels * 4, base_channels * 2, condition_dim
        )
        self.up2 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            4,
            stride=2,
            padding=1,
        )
        self.up_block2 = ConditionalResBlock(
            base_channels * 2, base_channels, condition_dim
        )
        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output_conv = nn.Conv2d(
            base_channels, img_channels, 3, padding=1
        )

    def forward(self, x, t, labels):
        condition = self.condition_mlp(
            self.time_embedding(t) + self.class_embedding(labels)
        )
        skip_input = self.input_conv(x)
        hidden = F.silu(self.down1(skip_input))
        skip_half = self.res1(hidden, condition)
        hidden = F.silu(self.down2(skip_half))
        hidden = self.res2(hidden, condition)
        hidden = self.mid1(hidden, condition)
        hidden = self.mid2(hidden, condition)
        hidden = F.silu(self.up1(hidden))
        hidden = self.up_block1(
            torch.cat((hidden, skip_half), dim=1), condition
        )
        hidden = F.silu(self.up2(hidden))
        hidden = self.up_block2(
            torch.cat((hidden, skip_input), dim=1), condition
        )
        hidden = F.silu(self.output_norm(hidden))
        return self.output_conv(hidden)

    def p_theta_step(self, x, t, labels, guidance_scale):
        batch_size = x.shape[0]
        t_tensor = torch.full(
            (batch_size,), t, dtype=torch.long, device=x.device
        )
        predicted_noise = self.predict_guided_noise(
            x, t_tensor, labels, guidance_scale
        )

        beta_t = self.betas[t].reshape(1, 1, 1, 1)
        alpha_t = self.alphas[t].reshape(1, 1, 1, 1)
        alpha_bar_t = self.alphas_bar[t].reshape(1, 1, 1, 1)
        if t > 0:
            alpha_bar_previous = self.alphas_bar[t - 1].reshape(
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
            mean = mean + torch.sqrt(posterior_variance) * torch.randn_like(
                x
            )
        return mean

    def predict_guided_noise(self, x, t, labels, guidance_scale):
        conditional_noise = self(x, t, labels)
        unconditional_labels = torch.full_like(labels, self.null_class)
        unconditional_noise = self(x, t, unconditional_labels)
        return unconditional_noise + guidance_scale * (
            conditional_noise - unconditional_noise
        )

    @torch.no_grad()
    def sample(self, labels, guidance_scale):
        self.eval()
        x_t = torch.randn(
            labels.shape[0],
            3,
            self.image_size,
            self.image_size,
            device=labels.device,
        )
        sequence = [x_t]
        for t in reversed(range(self.num_steps)):
            x_t = self.p_theta_step(x_t, t, labels, guidance_scale)
            sequence.append(x_t)
        return sequence

    @torch.no_grad()
    def sample_ddim(self, labels, guidance_scale, num_sample_steps=100):
        self.eval()
        x_t = torch.randn(
            labels.shape[0],
            3,
            self.image_size,
            self.image_size,
            device=labels.device,
        )
        sequence = [x_t]
        timesteps = np.linspace(
            self.num_steps - 1, 0, num_sample_steps, dtype=int
        )
        for index, t in enumerate(timesteps):
            t_tensor = torch.full(
                (labels.shape[0],),
                int(t),
                dtype=torch.long,
                device=labels.device,
            )
            predicted_noise = self.predict_guided_noise(
                x_t, t_tensor, labels, guidance_scale
            )
            alpha_bar_t = self.alphas_bar[int(t)].reshape(1, 1, 1, 1)
            predicted_x0 = (
                x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise
            ) / torch.sqrt(alpha_bar_t)
            predicted_x0 = predicted_x0.clamp(-1.0, 1.0)

            if index + 1 < len(timesteps):
                previous_t = int(timesteps[index + 1])
                alpha_bar_previous = self.alphas_bar[
                    previous_t
                ].reshape(1, 1, 1, 1)
            else:
                alpha_bar_previous = torch.ones_like(alpha_bar_t)

            x_t = (
                torch.sqrt(alpha_bar_previous) * predicted_x0
                + torch.sqrt(1 - alpha_bar_previous) * predicted_noise
            )
            sequence.append(x_t)
        return sequence

    def set_image_size(self, image_size):
        self.image_size = image_size


def diffusion_loss_fn(
    model,
    images,
    labels,
    label_dropout=0.0,
):
    batch_size = images.shape[0]
    t = torch.randint(
        0, model.num_steps, (batch_size,), device=images.device
    )
    noise = torch.randn_like(images)
    alpha_bar_t = model.alphas_bar[t].reshape(-1, 1, 1, 1)
    noisy_images = (
        torch.sqrt(alpha_bar_t) * images
        + torch.sqrt(1 - alpha_bar_t) * noise
    )

    training_labels = labels
    if label_dropout > 0:
        dropped = torch.rand(batch_size, device=images.device) < label_dropout
        training_labels = labels.clone()
        training_labels[dropped] = model.null_class

    predicted_noise = model(noisy_images, t, training_labels)
    return F.mse_loss(predicted_noise, noise)


def update_ema(ema_state, model_state, decay):
    with torch.no_grad():
        for key, value in model_state.items():
            if value.dtype.is_floating_point:
                ema_state[key].mul_(decay).add_(
                    value.detach(), alpha=1 - decay
                )
            else:
                ema_state[key].copy_(value)


def evaluate(model, dataloader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device).float() * 2 - 1
            labels = labels.to(device)
            loss = diffusion_loss_fn(model, images, labels)
            losses.append(loss.item())
    return float(np.mean(losses))


def image_from_tensor(image):
    return (
        (image.detach().clamp(-1, 1) + 1)
        / 2
    ).permute(1, 2, 0).cpu().numpy()


def save_visualizations(
    model,
    class_names,
    guidance_scale,
    vis_root,
):
    samples_per_class = 4
    labels = torch.arange(len(class_names), device=next(model.parameters()).device)
    labels = labels.repeat_interleave(samples_per_class)
    sequence = model.sample_ddim(
        labels,
        guidance_scale,
        num_sample_steps=100,
    )
    final_images = sequence[-1]

    fig, axes = plt.subplots(
        len(class_names),
        samples_per_class,
        figsize=(samples_per_class * 2.2, len(class_names) * 2.2),
    )
    for class_index, class_name in enumerate(class_names):
        for sample_index in range(samples_per_class):
            row = class_index
            column = sample_index
            flat_index = class_index * samples_per_class + sample_index
            ax = axes[row, column]
            ax.imshow(image_from_tensor(final_images[flat_index]))
            ax.axis("off")
            if column == 0:
                ax.set_ylabel(
                    class_name,
                    rotation=0,
                    labelpad=42,
                    fontsize=11,
                    va="center",
                )
    fig.suptitle(
        "Class-conditional samples with classifier-free guidance",
        fontsize=14,
    )
    fig.tight_layout()
    samples_path = os.path.join(vis_root, "diffusion_conditional_samples.png")
    fig.savefig(samples_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    snapshot_steps = np.linspace(
        0, len(sequence) - 1, num=9, dtype=int
    )
    ddim_timesteps = np.linspace(
        model.num_steps - 1, 0, len(sequence) - 1, dtype=int
    )
    fig, axes = plt.subplots(
        len(class_names),
        len(snapshot_steps),
        figsize=(len(snapshot_steps) * 1.7, len(class_names) * 1.9),
    )
    for class_index, class_name in enumerate(class_names):
        flat_index = class_index * samples_per_class
        for column, sequence_index in enumerate(snapshot_steps):
            ax = axes[class_index, column]
            ax.imshow(image_from_tensor(sequence[sequence_index][flat_index]))
            ax.axis("off")
            if sequence_index == 0:
                remaining_step = model.num_steps
            else:
                remaining_step = ddim_timesteps[int(sequence_index) - 1]
            ax.set_title(f"t={remaining_step}", fontsize=8)
            if column == 0:
                ax.set_ylabel(
                    class_name,
                    rotation=0,
                    labelpad=42,
                    fontsize=11,
                    va="center",
                )
    fig.suptitle(
        "Reverse diffusion process: noise to class-conditioned samples",
        fontsize=14,
    )
    fig.tight_layout()
    process_path = os.path.join(
        vis_root, "diffusion_conditional_denoising.png"
    )
    fig.savefig(process_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return samples_path, process_path


def save_training_curve(train_losses, validation_epochs, validation_losses, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        np.arange(1, len(train_losses) + 1),
        train_losses,
        label="Train loss",
        linewidth=1.8,
    )
    ax.plot(
        validation_epochs,
        validation_losses,
        label="Validation loss",
        marker="o",
        linewidth=1.6,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Noise prediction MSE")
    ax.set_title("Class-conditional DDPM training")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    device = get_device()
    print(f"Using device: {device}")

    data_root = "./flowers"
    model_root = "./model"
    log_root = "./log"
    vis_root = "./vis"
    os.makedirs(model_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)
    os.makedirs(vis_root, exist_ok=True)

    class_names = datasets.ImageFolder(root=data_root).classes
    training_loader, validation_loader = create_flower_dataloaders(
        args.batch_size,
        data_root,
        args.image_size,
        args.image_size,
    )

    model = ConditionalDiffuser(
        num_steps=args.num_steps,
        num_classes=len(class_names),
    ).to(device)
    model.set_image_size(args.image_size)
    ema_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    ema_model = ConditionalDiffuser(
        num_steps=args.num_steps,
        num_classes=len(class_names),
    ).to(device)
    ema_model.set_image_size(args.image_size)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=1e-5,
    )

    train_losses = []
    validation_epochs = []
    validation_losses = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_path = os.path.join(
        model_root, "BestModel_diffusion_conditional.pkl"
    )
    compatible_path = os.path.join(
        model_root, "Best_diffusion_conditional.pth"
    )
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for images, labels in training_loader:
            images = images.to(device).float() * 2 - 1
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = diffusion_loss_fn(
                model,
                images,
                labels,
                label_dropout=args.label_dropout,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            update_ema(ema_state, model.state_dict(), args.ema_decay)
            epoch_losses.append(loss.item())

        scheduler.step()
        train_loss = float(np.mean(epoch_losses))
        train_losses.append(train_loss)

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            ema_model.load_state_dict(ema_state)
            validation_loss = evaluate(
                ema_model, validation_loader, device
            )
            validation_epochs.append(epoch)
            validation_losses.append(validation_loss)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                torch.save(ema_state, best_path)
                torch.save(ema_state, compatible_path)
            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"train={train_loss:.5f} | valid={validation_loss:.5f} | "
                f"best={best_validation_loss:.5f}"
            )
        else:
            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"train={train_loss:.5f}"
            )

    elapsed_minutes = (time.time() - start_time) / 60
    ema_model.load_state_dict(torch.load(best_path, map_location=device))

    samples_path, process_path = save_visualizations(
        ema_model,
        class_names,
        args.guidance_scale,
        vis_root,
    )
    curve_path = os.path.join(
        vis_root, "diffusion_conditional_training_curve.png"
    )
    save_training_curve(
        train_losses,
        validation_epochs,
        validation_losses,
        curve_path,
    )

    metrics = {
        "dataset": data_root,
        "class_names": class_names,
        "train_images": len(training_loader.dataset),
        "validation_images": len(validation_loader.dataset),
        "epochs": args.epochs,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_steps": args.num_steps,
        "learning_rate": args.learning_rate,
        "guidance_scale": args.guidance_scale,
        "ema_decay": args.ema_decay,
        "label_dropout": args.label_dropout,
        "sampler": "DDIM",
        "sample_steps": 100,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "final_train_loss": train_losses[-1],
        "elapsed_minutes": elapsed_minutes,
        "samples_figure": samples_path,
        "denoising_figure": process_path,
        "training_curve": curve_path,
    }
    metrics_path = os.path.join(
        log_root, "diffusion_conditional_metrics.json"
    )
    with open(metrics_path, "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    print(f"Saved model to {best_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Elapsed time: {elapsed_minutes:.2f} minutes")


if __name__ == "__main__":
    main()
