# High-Resolution Conditional DDPM

This experiment moves the Diffusion bonus from the local 4 GB GPU to Kaggle
Tesla T4.

## Setup

- Input: private Kaggle dataset `yuanzelei/mllab2026-flowers-subset`
- Resolution: 64 x 64
- Model: conditional residual U-Net with low-resolution self-attention
- Training: AdamW, mixed precision, cosine learning-rate decay, EMA
- Sampling: classifier-free guidance and 100-step deterministic DDIM
- Evaluation: held-out flower classifier and generated-label consistency

## Files

- `train_highres_diffusion.py`: readable source code
- `highres_diffusion.ipynb`: Kaggle notebook payload
- `kernel-metadata.json`: Kaggle kernel configuration

## Kaggle API

Use the local `kaggle-rest-api` skill helper:

```powershell
& "$HOME\.codex\skills\kaggle-rest-api\scripts\kaggle.ps1" push `
  .\kaggle\highres_diffusion --timeout 21600
```

The normalized Kaggle slug is
`mllab2026-high-resolution-diffusion`.
