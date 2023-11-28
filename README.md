# DiffTAD
Official PyTorch implementation for "DiffTAD: Denoising Diffusion Probabilistic Models for Vehicle Trajectory Anomaly Detection". The source code will be made publicly available upon publication.

# 📌 Pipeline Overview
![image](https://github.com/Psychic-DL/DiffTAD/blob/ea7bcbc196c7aa552e561ec038a6b598cd9b5609/figs/pipeline.png)

# 🎯 Highlights
- A new framework for vehicle trajectory anomaly detection that formalizes this problem as a noisy-to-normal paradigm.
- Reconstructing near-normal trajectories from trajectories corrupted by Gaussian noise and detecting anomalies by comparing the differences between the query trajectories and their reconstructions.
- Transformer-based temporal and spatial encoders are integrated to model the temporal dependencies and spatial interactions of vehicles in the diffusion model.
- The interval sampling strategy accelerates the inference process of diffusion models.
