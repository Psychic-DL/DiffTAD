# DiffTAD
Official PyTorch implementation for "DiffTAD: Denoising Diffusion Probabilistic Models for Vehicle Trajectory Anomaly Detection". The source code will be made publicly available upon publication.

## 🎯 Motivation
Vehicle trajectory anomaly detection plays an essential role in the fields of traffic video surveillance, autonomous driving navigation, and taxi fraud detection. Deep generative models have been shown to be promising solutions for anomaly detection, avoiding the costs involved in manual labeling. However, existing popular generative models such as Generative Adversarial Networks (GANs) and Variational AutoEncoders (VAEs) are often plagued by training instability, mode collapse, and poor sample quality. To resolve the dilemma, we present DiffTAD, a novel vehicle trajectory anomaly detection framework based on the emerging diffusion models. DiffTAD formalizes anomaly detection as a noisy-to-normal process that progressively adds noise to the vehicle trajectory until the path is corrupted to pure Gaussian noise. The core idea of our framework is to devise deep neural networks to learn the reverse of the diffusion process and to detect anomalies by comparing the difference between a query trajectory and its reconstruction. DiffTAD is a parameterized Markov chain trained with variational inference and allows the mean square error to optimize the reweighted variational lower bound. In addition, DiffTAD integrates decoupled Transformer-based temporal and spatial encoders to model the temporal dependencies and spatial interactions among vehicles in the diffusion models.

## 📌 Pipeline Overview
![image](https://github.com/Psychic-DL/DiffTAD/blob/ea7bcbc196c7aa552e561ec038a6b598cd9b5609/figs/pipeline.png)

## 🚀 Highlights
- A new framework for vehicle trajectory anomaly detection that formalizes this problem as a noisy-to-normal paradigm.
- Reconstructing near-normal trajectories from trajectories corrupted by Gaussian noise and detecting anomalies by comparing the differences between the query trajectories and their reconstructions.
- Transformer-based temporal and spatial encoders are integrated to model the temporal dependencies and spatial interactions of vehicles in the diffusion model.
- The interval sampling strategy accelerates the inference process of diffusion models.

## 🔧 Model
![image](https://github.com/Psychic-DL/DiffTAD/blob/9ce8b7d144e4355d919c2d9fb0d419c67777adb4/figs/difftad.png)

## 📜 Algorithm Pseudo-Code

## 🗂️ Datasets
For the TRAFFIC, CROSS, and Syntra datasets, you can contact me by email for a copy if needed (xdchaonengli@163.com). To obtain the MAAD highway dataset, please follow the instructions from the [maad_highway](https://github.com/againerju/maad_highway) Github repository to request the access link.

## 🔑 Evaluation
