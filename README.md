# Final TiGraph–SpaFormer (Paper-Clean, Perf-Safe)

This repository contains a GitHub-ready training script for the FINAL TiGraph–SpaFormer variant:
- Temporal: Multi-Periodic + Noise Embedding + FFT(FAE) + per-sample CPD pooling + residual-to-last
- Spatial: OT-Proto-S + MS-PARA
- Loss: Huber + Q-loss + freq-weighted + TV regularization on A-field

## Requirements
- Python 3.10+
- PyTorch (CUDA optional)
- numpy

Example:
```bash
pip install torch numpy
