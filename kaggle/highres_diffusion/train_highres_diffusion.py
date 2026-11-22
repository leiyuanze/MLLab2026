# %% [markdown]
# # MLlab2026 High-Resolution Class-Conditional DDPM
#
# This notebook trains a 64x64 class-conditional diffusion model on Kaggle T4,
# uses EMA weights, and evaluates generation with DDIM plus a held-out flower
# classifier.

# %%
import json
import math
import os
import random
import time
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from tqdm.auto import tqdm


SMOKE_TEST = os.getenv("SMOKE_TEST", "0") == "1"
DATA_ROOT = Path(
    os.getenv(
        "DATA_ROOT",
        "/kaggle/input",
    )
)
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", "/kaggle/working"))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "32" if SMOKE_TEST else "64"))
NUM_STEPS = int(os.getenv("NUM_STEPS", "50" if SMOKE_TEST else "1000"))
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", "1" if SMOKE_TEST else "300"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4" if SMOKE_TEST else "32"))
BASE_CHANNELS = int(
    os.getenv("BASE_CHANNELS", "32" if SMOKE_TEST else "64")
)
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
EMA_DECAY = float(os.getenv("EMA_DECAY", "0.9995"))
LABEL_DROPOUT = 0.1
GUIDANCE_SCALE = 2.5
DDIM_STEPS = int(os.getenv("DDIM_STEPS", "20" if SMOKE_TEST else "100"))
MAX_TRAIN_MINUTES = float(os.getenv("MAX_TRAIN_MINUTES", "210"))
EVAL_EVERY = 5 if SMOKE_TEST else 10

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
if device.type == "cuda":
    print("gpu:", torch.cuda.get_device_name(0))
    print("gpu_memory_gb:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))


# %%
def find_imagefolder_root(root):
    root = Path(root)
    search_roots = [root]
    archives = list(root.rglob("*.zip"))
    if archives:
        if Path("/kaggle/input").exists():
            extraction_root = Path("/tmp/mllab2026_dataset_extracted")
        else:
            extraction_root = OUTPUT_ROOT / "dataset_extracted"
        extraction_root.mkdir(parents=True, exist_ok=True)
        for archive_path in archives:
            archive_output = extraction_root / archive_path.stem
            archive_output.mkdir(parents=True, exist_ok=True)
            print("extracting:", archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(archive_output)
        search_roots.append(extraction_root)

    candidates = []
    for search_root in search_roots:
        candidates.append(search_root)
        candidates.extend(path.parent for path in search_root.rglob("*"))
    best = None
    best_classes = 0
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        class_dirs = [
            path
            for path in candidate.iterdir()
            if path.is_dir()
            and any(
                child.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                for child in path.iterdir()
                if child.is_file()
            )
        ]
        if len(class_dirs) > best_classes:
            best = candidate
            best_classes = len(class_dirs)
    if best is None or best_classes < 2:
        raise FileNotFoundError(f"No ImageFolder dataset found below {root}")
    return best


imagefolder_root = find_imagefolder_root(DATA_ROOT)
print("imagefolder_root:", imagefolder_root)

transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
    ]
)
source_dataset = datasets.ImageFolder(
    root=str(imagefolder_root),
    transform=transform,
)
class_names = source_dataset.classes
num_classes = len(class_names)
print("classes:", class_names)

images = []
labels = []
for image, label in tqdm(source_dataset, desc="cache images"):
    images.append(image)
    labels.append(label)
images = torch.stack(images)
labels = torch.tensor(labels, dtype=torch.long)

if SMOKE_TEST:
    keep_indices = []
    for class_index in range(num_classes):
        class_indices = torch.where(labels == class_index)[0][:32]
        keep_indices.append(class_indices)
    keep_indices = torch.cat(keep_indices)
    images = images[keep_indices]
    labels = labels[keep_indices]

cached_dataset = TensorDataset(images, labels)
train_size = int(0.8 * len(cached_dataset))
validation_size = len(cached_dataset) - train_size
train_dataset, validation_dataset = random_split(
    cached_dataset,
    [train_size, validation_size],
    generator=torch.Generator().manual_seed(0),
)

loader_workers = 0 if SMOKE_TEST else 2
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=loader_workers,
    pin_memory=device.type == "cuda",
    persistent_workers=loader_workers > 0,
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=loader_workers,
    pin_memory=device.type == "cuda",
    persistent_workers=loader_workers > 0,
)
print("train_images:", len(train_dataset))
print("validation_images:", len(validation_dataset))


# %%
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
        hidden = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.condition(condition).chunk(2, dim=1)
        hidden = hidden * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return (hidden + self.shortcut(x)) / math.sqrt(2)


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        qkv = self.qkv(self.norm(x))
        query, key, value = qkv.chunk(3, dim=1)
        query = query.reshape(batch, channels, height * width).transpose(1, 2)
        key = key.reshape(batch, channels, height * width)
        attention = torch.softmax(
            query @ key / math.sqrt(channels),
            dim=-1,
        )
        value = value.reshape(batch, channels, height * width)
        hidden = (value @ attention.transpose(1, 2)).reshape(
            batch,
            channels,
            height,
            width,
        )
        return x + self.output(hidden)


class ConditionalDiffuser(nn.Module):
    def __init__(
        self,
        num_steps,
        num_classes,
        image_size,
        base_channels=64,
        condition_dim=256,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.null_class = num_classes
        self.image_size = image_size

        offset = 0.008
        timesteps = torch.linspace(0, num_steps, num_steps + 1)
        alpha_bar = torch.cos(
            ((timesteps / num_steps + offset) / (1 + offset)) * math.pi / 2
        ) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = torch.clamp(
            1 - alpha_bar[1:] / alpha_bar[:-1],
            min=1e-4,
            max=0.999,
        )
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("alphas_bar", torch.cumprod(1 - betas, dim=0))

        self.time_embedding = nn.Embedding(num_steps, condition_dim)
        self.class_embedding = nn.Embedding(num_classes + 1, condition_dim)
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 4

        self.input_conv = nn.Conv2d(3, c1, 3, padding=1)
        self.down1 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)
        self.res1 = ConditionalResBlock(c2, c2, condition_dim)
        self.down2 = nn.Conv2d(c2, c3, 4, stride=2, padding=1)
        self.res2 = ConditionalResBlock(c3, c3, condition_dim)
        self.down3 = nn.Conv2d(c3, c4, 4, stride=2, padding=1)
        self.res3 = ConditionalResBlock(c4, c4, condition_dim)
        self.mid1 = ConditionalResBlock(c4, c4, condition_dim)
        self.attention = SelfAttentionBlock(c4)
        self.mid2 = ConditionalResBlock(c4, c4, condition_dim)

        self.up1 = nn.ConvTranspose2d(c4, c3, 4, stride=2, padding=1)
        self.up_block1 = ConditionalResBlock(c3 + c3, c3, condition_dim)
        self.up2 = nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)
        self.up_block2 = ConditionalResBlock(c2 + c2, c2, condition_dim)
        self.up3 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)
        self.up_block3 = ConditionalResBlock(c1 + c1, c1, condition_dim)
        self.output_norm = nn.GroupNorm(8, c1)
        self.output_conv = nn.Conv2d(c1, 3, 3, padding=1)

    def forward(self, x, t, labels):
        condition = self.condition_mlp(
            self.time_embedding(t) + self.class_embedding(labels)
        )
        skip1 = self.input_conv(x)
        hidden = self.res1(F.silu(self.down1(skip1)), condition)
        skip2 = hidden
        hidden = self.res2(F.silu(self.down2(skip2)), condition)
        skip3 = hidden
        hidden = self.res3(F.silu(self.down3(skip3)), condition)
        hidden = self.mid1(hidden, condition)
        hidden = self.attention(hidden)
        hidden = self.mid2(hidden, condition)
        hidden = self.up_block1(
            torch.cat((F.silu(self.up1(hidden)), skip3), dim=1),
            condition,
        )
        hidden = self.up_block2(
            torch.cat((F.silu(self.up2(hidden)), skip2), dim=1),
            condition,
        )
        hidden = self.up_block3(
            torch.cat((F.silu(self.up3(hidden)), skip1), dim=1),
            condition,
        )
        return self.output_conv(F.silu(self.output_norm(hidden)))

    def predict_guided_noise(self, x, t, labels, guidance_scale):
        conditional = self(x, t, labels)
        unconditional_labels = torch.full_like(labels, self.null_class)
        unconditional = self(x, t, unconditional_labels)
        return unconditional + guidance_scale * (conditional - unconditional)

    @torch.no_grad()
    def sample_ddim(self, labels, guidance_scale, sample_steps):
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
            self.num_steps - 1,
            0,
            sample_steps,
            dtype=int,
        )
        for index, timestep in enumerate(timesteps):
            t = torch.full(
                (labels.shape[0],),
                int(timestep),
                dtype=torch.long,
                device=labels.device,
            )
            predicted_noise = self.predict_guided_noise(
                x_t,
                t,
                labels,
                guidance_scale,
            )
            alpha_bar_t = self.alphas_bar[int(timestep)].reshape(1, 1, 1, 1)
            predicted_x0 = (
                x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise
            ) / torch.sqrt(alpha_bar_t)
            predicted_x0 = predicted_x0.clamp(-1, 1)

            if index + 1 < len(timesteps):
                previous_alpha_bar = self.alphas_bar[
                    int(timesteps[index + 1])
                ].reshape(1, 1, 1, 1)
            else:
                previous_alpha_bar = torch.ones_like(alpha_bar_t)
            x_t = (
                torch.sqrt(previous_alpha_bar) * predicted_x0
                + torch.sqrt(1 - previous_alpha_bar) * predicted_noise
            )
            sequence.append(x_t)
        return sequence


model = ConditionalDiffuser(
    NUM_STEPS,
    num_classes,
    IMAGE_SIZE,
    BASE_CHANNELS,
).to(device)
ema_model = ConditionalDiffuser(
    NUM_STEPS,
    num_classes,
    IMAGE_SIZE,
    BASE_CHANNELS,
).to(device)
print("model_parameters:", sum(parameter.numel() for parameter in model.parameters()))


# %%
def diffusion_loss_fn(model, images, labels):
    batch_size = images.shape[0]
    t = torch.randint(0, NUM_STEPS, (batch_size,), device=images.device)
    noise = torch.randn_like(images)
    alpha_bar_t = model.alphas_bar[t].reshape(-1, 1, 1, 1)
    noisy_images = (
        torch.sqrt(alpha_bar_t) * images
        + torch.sqrt(1 - alpha_bar_t) * noise
    )
    dropped = torch.rand(batch_size, device=images.device) < LABEL_DROPOUT
    training_labels = labels.clone()
    training_labels[dropped] = model.null_class
    predicted_noise = model(noisy_images, t, training_labels)
    return F.mse_loss(predicted_noise, noise)


def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_parameter, parameter in zip(
            ema_model.parameters(),
            model.parameters(),
        ):
            ema_parameter.mul_(decay).add_(parameter.detach(), alpha=1 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def evaluate(model, dataloader):
    model.eval()
    losses = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True).float() * 2 - 1
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                loss = diffusion_loss_fn(model, images, labels)
            losses.append(loss.item())
    return float(np.mean(losses))


optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=max(MAX_EPOCHS, 1),
    eta_min=1e-5,
)
scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
ema_model.load_state_dict(model.state_dict())

train_losses = []
validation_epochs = []
validation_losses = []
best_validation_loss = float("inf")
best_epoch = 0
best_state = None
training_start = time.time()

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    epoch_losses = []
    progress = tqdm(train_loader, desc=f"epoch {epoch}/{MAX_EPOCHS}")
    for images, labels in progress:
        images = images.to(device, non_blocking=True).float() * 2 - 1
        labels = labels.to(device, non_blocking=True)
        flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
        if flip_mask.any():
            images[flip_mask] = torch.flip(images[flip_mask], dims=[3])

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            loss = diffusion_loss_fn(model, images, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        update_ema(ema_model, model, EMA_DECAY)
        epoch_losses.append(loss.item())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    train_loss = float(np.mean(epoch_losses))
    train_losses.append(train_loss)

    if epoch % EVAL_EVERY == 0 or epoch == MAX_EPOCHS:
        validation_loss = evaluate(ema_model, validation_loader)
        validation_epochs.append(epoch)
        validation_losses.append(validation_loss)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().half()
                if value.is_floating_point()
                else value.detach().cpu()
                for key, value in ema_model.state_dict().items()
            }
            torch.save(best_state, OUTPUT_ROOT / "BestModel_diffusion_highres.pkl")
            torch.save(best_state, OUTPUT_ROOT / "Best_diffusion_highres.pth")

    elapsed_minutes = (time.time() - training_start) / 60
    print(
        f"epoch={epoch} train={train_loss:.5f} "
        f"best_valid={best_validation_loss:.5f} "
        f"elapsed_min={elapsed_minutes:.2f}"
    )
    if elapsed_minutes >= MAX_TRAIN_MINUTES:
        print("Stopping at the configured training-time budget.")
        break


# %%
if best_state is None:
    raise RuntimeError("No EMA checkpoint was created")
model.load_state_dict(best_state)
model.eval()


def image_from_tensor(image):
    return (
        (image.detach().clamp(-1, 1) + 1) / 2
    ).permute(1, 2, 0).cpu().numpy()


samples_per_class = 3 if SMOKE_TEST else 6
sample_labels = torch.arange(num_classes, device=device).repeat_interleave(
    samples_per_class
)
sample_sequence = model.sample_ddim(
    sample_labels,
    GUIDANCE_SCALE,
    DDIM_STEPS,
)
final_samples = sample_sequence[-1]

fig, axes = plt.subplots(
    num_classes,
    samples_per_class,
    figsize=(samples_per_class * 2.1, num_classes * 2.2),
)
if num_classes == 1:
    axes = np.array([axes])
for class_index, class_name in enumerate(class_names):
    for sample_index in range(samples_per_class):
        flat_index = class_index * samples_per_class + sample_index
        ax = axes[class_index, sample_index]
        ax.imshow(image_from_tensor(final_samples[flat_index]))
        ax.axis("off")
        if sample_index == 0:
            ax.set_ylabel(
                class_name,
                rotation=0,
                labelpad=48,
                fontsize=11,
                va="center",
            )
fig.suptitle("64x64 class-conditional DDIM samples", fontsize=15)
fig.tight_layout()
fig.savefig(OUTPUT_ROOT / "samples_highres.png", dpi=240, bbox_inches="tight")
plt.close(fig)

snapshot_indices = np.linspace(
    0,
    len(sample_sequence) - 1,
    num=9,
    dtype=int,
)
ddim_timesteps = np.linspace(
    NUM_STEPS - 1,
    0,
    len(sample_sequence) - 1,
    dtype=int,
)
fig, axes = plt.subplots(
    num_classes,
    len(snapshot_indices),
    figsize=(len(snapshot_indices) * 1.8, num_classes * 2.0),
)
if num_classes == 1:
    axes = np.array([axes])
for class_index, class_name in enumerate(class_names):
    flat_index = class_index * samples_per_class
    for column, sequence_index in enumerate(snapshot_indices):
        ax = axes[class_index, column]
        ax.imshow(
            image_from_tensor(sample_sequence[sequence_index][flat_index])
        )
        ax.axis("off")
        remaining_t = (
            NUM_STEPS
            if sequence_index == 0
            else ddim_timesteps[int(sequence_index) - 1]
        )
        ax.set_title(f"t={remaining_t}", fontsize=8)
        if column == 0:
            ax.set_ylabel(
                class_name,
                rotation=0,
                labelpad=48,
                fontsize=11,
                va="center",
            )
fig.suptitle("64x64 DDIM reverse diffusion", fontsize=15)
fig.tight_layout()
fig.savefig(
    OUTPUT_ROOT / "denoising_highres.png",
    dpi=240,
    bbox_inches="tight",
)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(np.arange(1, len(train_losses) + 1), train_losses, label="Train")
ax.plot(validation_epochs, validation_losses, marker="o", label="Validation EMA")
ax.set_xlabel("Epoch")
ax.set_ylabel("Noise prediction MSE")
ax.set_title("64x64 conditional DDPM training")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(
    OUTPUT_ROOT / "training_curve_highres.png",
    dpi=220,
    bbox_inches="tight",
)
plt.close(fig)


# %%
class FlowerClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


classifier = FlowerClassifier(num_classes).to(device)
classifier_optimizer = AdamW(
    classifier.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)
classifier_epochs = 2 if SMOKE_TEST else 15
classifier_scaler = torch.amp.GradScaler(
    "cuda",
    enabled=device.type == "cuda",
)

for classifier_epoch in range(classifier_epochs):
    classifier.train()
    for classifier_images, classifier_labels in train_loader:
        classifier_images = classifier_images.to(
            device,
            non_blocking=True,
        )
        classifier_labels = classifier_labels.to(
            device,
            non_blocking=True,
        )
        classifier_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = classifier(classifier_images)
            classifier_loss = F.cross_entropy(logits, classifier_labels)
        classifier_scaler.scale(classifier_loss).backward()
        classifier_scaler.step(classifier_optimizer)
        classifier_scaler.update()

classifier.eval()
correct = 0
total = 0
with torch.no_grad():
    for classifier_images, classifier_labels in validation_loader:
        classifier_images = classifier_images.to(device)
        classifier_labels = classifier_labels.to(device)
        predictions = classifier(classifier_images).argmax(dim=1)
        correct += (predictions == classifier_labels).sum().item()
        total += classifier_labels.numel()
classifier_validation_accuracy = correct / total

with torch.no_grad():
    generated_predictions = classifier(final_samples.to(device)).argmax(dim=1)
conditional_consistency = (
    generated_predictions == sample_labels
).float().mean().item()


# %%
elapsed_minutes = (time.time() - training_start) / 60
metrics = {
    "device": str(device),
    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    "classes": class_names,
    "train_images": len(train_dataset),
    "validation_images": len(validation_dataset),
    "image_size": IMAGE_SIZE,
    "num_steps": NUM_STEPS,
    "ddim_steps": DDIM_STEPS,
    "batch_size": BATCH_SIZE,
    "base_channels": BASE_CHANNELS,
    "max_epochs": MAX_EPOCHS,
    "completed_epochs": len(train_losses),
    "best_epoch": best_epoch,
    "best_validation_loss": best_validation_loss,
    "final_train_loss": train_losses[-1],
    "guidance_scale": GUIDANCE_SCALE,
    "label_dropout": LABEL_DROPOUT,
    "ema_decay": EMA_DECAY,
    "classifier_validation_accuracy": classifier_validation_accuracy,
    "conditional_consistency": conditional_consistency,
    "elapsed_minutes": elapsed_minutes,
}
(OUTPUT_ROOT / "metrics_highres.json").write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)
print(json.dumps(metrics, indent=2))
