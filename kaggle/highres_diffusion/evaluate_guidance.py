import argparse
import ast
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-class", type=int, default=12)
    parser.add_argument("--classifier-epochs", type=int, default=15)
    return parser.parse_args()


def extract_model_classes(source_path):
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in {
            "ConditionalResBlock",
            "SelfAttentionBlock",
            "ConditionalDiffuser",
        }:
            selected.append(node)
    namespace = {"__name__": "highres_model_extract"}
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["ConditionalDiffuser"]


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


def main():
    args = parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize(
                (64, 64),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(args.data_root, transform=transform)
    class_names = dataset.classes
    images = []
    labels = []
    for image, label in tqdm(dataset, desc="cache"):
        images.append(image)
        labels.append(label)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    cached = TensorDataset(images, labels)
    train_size = int(0.8 * len(cached))
    train_dataset, validation_dataset = random_split(
        cached,
        [train_size, len(cached) - train_size],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    classifier = FlowerClassifier(len(class_names)).to(device)
    optimizer = AdamW(
        classifier.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(args.classifier_epochs):
        classifier.train()
        losses = []
        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = classifier(batch_images)
                loss = F.cross_entropy(logits, batch_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        print(f"classifier epoch={epoch + 1} loss={np.mean(losses):.5f}")

    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_images, batch_labels in validation_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            predictions = classifier(batch_images).argmax(dim=1)
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.numel()
    validation_accuracy = correct / total

    model_class = extract_model_classes(Path(__file__).with_name(
        "train_highres_diffusion.py"
    ))
    model = model_class(1000, len(class_names), 64, 64).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    results = {
        "classifier_validation_accuracy": validation_accuracy,
        "guidance": {},
    }
    for guidance_scale in (2.5, 4.0, 6.0):
        sample_labels = torch.arange(
            len(class_names), device=device
        ).repeat_interleave(args.samples_per_class)
        sequence = model.sample_ddim(
            sample_labels,
            guidance_scale,
            100,
        )
        generated = sequence[-1]
        with torch.no_grad():
            predictions = classifier(generated.to(device)).argmax(dim=1)
        consistency = (
            predictions == sample_labels
        ).float().mean().item()
        per_class = {}
        for class_index, class_name in enumerate(class_names):
            class_mask = sample_labels == class_index
            per_class[class_name] = (
                predictions[class_mask] == class_index
            ).float().mean().item()
        results["guidance"][str(guidance_scale)] = {
            "conditional_consistency": consistency,
            "per_class_consistency": per_class,
        }

        fig, axes = plt.subplots(
            len(class_names),
            args.samples_per_class,
            figsize=(args.samples_per_class * 1.8, len(class_names) * 1.9),
        )
        for class_index in range(len(class_names)):
            for sample_index in range(args.samples_per_class):
                flat_index = (
                    class_index * args.samples_per_class + sample_index
                )
                image = (
                    (generated[flat_index].clamp(-1, 1) + 1) / 2
                ).permute(1, 2, 0).cpu().numpy()
                axes[class_index, sample_index].imshow(image)
                axes[class_index, sample_index].axis("off")
        fig.suptitle(f"guidance={guidance_scale:.1f}")
        fig.tight_layout()
        fig.savefig(
            output_root / f"guidance_{guidance_scale:.1f}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)
        print(f"guidance={guidance_scale}: consistency={consistency:.3f}")

    (output_root / "guidance_metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
