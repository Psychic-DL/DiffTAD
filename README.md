# DiffTAD
Official PyTorch implementation for "DiffTAD: Denoising Diffusion Probabilistic Models for Vehicle Trajectory Anomaly Detection". The source code will be made publicly available upon publication.

# 📌 Pipeline Overview
![image](https://github.com/Psychic-DL/DiffTAD/blob/ea7bcbc196c7aa552e561ec038a6b598cd9b5609/figs/pipeline.png)

# 🎯 Abstract
Vehicle trajectory anomaly detection plays an essential role in the fields of traffic video surveillance, autonomous driving navigation, and taxi fraud detection. Deep generative models have been shown to be promising solutions for anomaly detection, avoiding the costs involved in manual labeling. However, existing popular generative models such as Generative Adversarial Networks (GANs) and Variational AutoEncoders (VAEs) are often plagued by training instability, mode collapse, and poor sample quality. To resolve the dilemma, we present DiffTAD, a novel vehicle trajectory anomaly detection framework based on the emerging diffusion models. DiffTAD formalizes anomaly detection as a noisy-to-normal process that progressively adds noise to the vehicle trajectory until the path is corrupted to pure Gaussian noise. The core idea of our framework is to devise deep neural networks to learn the reverse of the diffusion process and to detect anomalies by comparing the difference between a query trajectory and its reconstruction. DiffTAD is a parameterized Markov chain trained with variational inference and allows the mean square error to optimize the reweighted variational lower bound. In addition, DiffTAD integrates decoupled Transformer-based temporal and spatial encoders to model the temporal dependencies and spatial interactions among vehicles in the diffusion models. Extensive experiments on real-world and synthetic trajectory datasets show that our DiffTAD outperforms the baselines and achieves state-of-the-art performance on multiple evaluation metrics.
