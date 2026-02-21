# Deep Generative Models

## 📌 Overview

This repository contains a comprehensive collection of implementations and research reports for the **Deep Generative Models** course at the University of Tehran. The projects explore the theoretical foundations and practical applications of modern generative architectures, ranging from Variational Autoencoders (VAEs) to Diffusion Models.

Each project includes a dedicated report, mathematical derivations, and a self-contained Jupyter Notebook implementation.

---

## 📂 Projects

| Project | Description | Key Topics |
| --- | --- | --- |
| **1. VAE & Disentanglement** | Analysis of latent variable independence using **-VAE** on the dSprites dataset. Includes custom implementations of the **MIG (Mutual Information Gap)** metric to quantify disentanglement. | `VAE` `Beta-VAE` `Disentanglement` `MIG Score` |
| **2. Normalizing Flows & GANs** | Implementation of **Masked Autoregressive Flows (MAF)** for industrial anomaly detection (MVTec AD) and **CycleGAN** for unpaired style transfer (Summer  Winter). | `Normalizing Flows` `MADE` `CycleGAN` `Anomaly Detection` |
| **3. Energy & Score-Based Models** | Exploration of implicit likelihood models. Features an **Energy-Based Model (EBM)** with a replay buffer and a **Noise Conditional Score Network (NCSN)** for conditional digit generation. | `EBM` `Langevin Dynamics` `Score Matching` `NCSN` |
| **4. Diffusion & Flow Matching** | Implementation of **DDPM & DDIM** from scratch for FashionMNIST generation, parameter-efficient fine-tuning of Stable Diffusion via **DreamBooth & LoRA**, and **Conditional Flow Matching** for synthesizing financial time series (SPY ETF). | `Diffusion` `DDPM/DDIM` `DreamBooth` `Flow Matching` |

*Note: Please refer to the individual project folders for detailed READMEs.*
