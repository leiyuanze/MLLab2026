# High-Resolution Conditional DDPM

This experiment moves the Diffusion bonus from the local 4 GB GPU to a
Kaggle GPU (Tesla T4 or P100).

## Setup

- Input: private Kaggle dataset `yuanzelei/mllab2026-flowers-subset`
- Resolution: 64 x 64
- Model: conditional residual U-Net with low-resolution self-attention
- Training: AdamW, mixed precision, cosine learning-rate decay, EMA
- Sampling: classifier-free guidance and 100-step deterministic DDIM
- Evaluation: held-out flower classifier and generated-label consistency

## Completed Run

- Kaggle-reported accelerator: Tesla T4, 15.64 GB
- Epochs: 300
- Best validation loss: 0.05733 at epoch 270
- Held-out classifier accuracy: 89.96%
- Guidance comparison: 2.5 -> 86.1%, 4.0 -> 86.1%, 6.0 -> 97.2%
- Final sampling guidance: 6.0

The uploaded Kaggle dataset is mounted under
`/kaggle/input/datasets/yuanzelei/mllab2026-flowers-subset`; the training
script searches all of `/kaggle/input` so it also works when that path changes.

## Files

- `train_highres_diffusion.py`: readable source code
- `highres_diffusion.ipynb`: Kaggle notebook payload
- `kernel-metadata.json`: Kaggle kernel configuration
- `evaluate_guidance.py`: guidance sweep and held-out classifier evaluation

## Kaggle API

Use the local `kaggle-rest-api` skill helper:

```powershell
& "$HOME\.codex\skills\kaggle-rest-api\scripts\kaggle.ps1" push `
  .\kaggle\highres_diffusion --timeout 21600
```

The latest normalized Kaggle slug is
`mllab2026-highres-p100-v2`.

Completed private notebook:
https://www.kaggle.com/code/yuanzelei/mllab2026-highres-p100-v2
