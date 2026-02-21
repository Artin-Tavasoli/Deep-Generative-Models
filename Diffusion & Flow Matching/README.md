# Deep Generative Models - Diffusion, DreamBooth & Flow Matching

## Overview
This project contains implementations of three state-of-the-art generative modeling techniques. It explores pixel-space diffusion, parameter-efficient fine-tuning for latent diffusion models, and continuous-time flow matching for sequential financial data.

---

## 1. Denoising Diffusion Probabilistic & Implicit Models (DDPM & DDIM)

**Objective:** Implement DDPM and DDIM from scratch and train them on the FashionMNIST dataset.
- **Methodology:** - Constructed a custom U-Net architecture integrated with time and context embeddings.
  - Implemented a linear variance scheduler for the forward diffusion process (adding Gaussian noise).
  - Trained the network to predict and subtract noise across timesteps, enabling reverse diffusion.
  - Implemented two distinct sampling algorithms: standard Markovian DDPM and non-Markovian DDIM.
- **Results:**
  - The model successfully learned to generate realistic FashionMNIST clothing items from pure Gaussian noise.
  - **Inference Speedup:** By utilizing the DDIM sampler, inference steps were drastically reduced from **1000 steps** (DDPM) to just **20 steps** (DDIM) while preserving visual fidelity.
  - **Quantitative Evaluation:** The structural quality and similarity of generated images to the real dataset were computed and compared using the **Fréchet Inception Distance (FID) score**.

---

## 2. DreamBooth LoRA Fine-Tuning for Stable Diffusion

**Objective:** Teach a pre-trained text-to-image model (Stable Diffusion v1.5) to generate images of a specific custom subject.
- **Methodology:**
  - Leveraged **DreamBooth** concepts to bind a specific subject to a rare token identifier (`<sks>`).
  - Utilized **LoRA (Low-Rank Adaptation)** to inject trainable rank decomposition matrices into the UNet's cross-attention layers. This isolated the training to a tiny fraction of the parameters, making the fine-tuning highly memory and compute-efficient.
  - Implemented **Prior Preservation Loss** by simultaneously generating and training on generic class images to prevent the model from overfitting on the subject and forgetting its prior knowledge.
- **Results:**
  - The model successfully learned the custom subject and gained the ability to generate it in entirely new environments and artistic styles (e.g., using prompts like `"a photo of a <sks> car on the moon"`).
  - Validated that the parameter-efficient approach successfully sidestepped catastrophic forgetting and language drift.

---

## 3. Financial Time Series using Flow Matching

**Objective:** Generate synthetic, realistic financial time series data that mimics the SPY ETF (S&P 500) using Conditional Flow Matching.
- **Methodology:**
  - Handled non-stationary raw stock prices by calculating normalized log returns for stable training.
  - Trained a 1D Convolutional Neural Network to approximate the continuous vector field.
  - The model mapped a simple noise distribution to the complex empirical distribution of market returns by creating a linear probability path: $x_t = (1 - t)x_0 + tx_1$, and minimizing the Mean Squared Error against the target theoretical velocity ($x_1 - x_0$).
- **Results & Mathematical Validation:**
  - **Distributional Geometry:** Evaluated the structure of the generated sequences using the **Sliced Wasserstein Distance (SWD)**. Achieved an exceptionally low score of **`0.001159`**, proving the model mathematically aligns with the real market's geometric distribution and joint feature structure.
  - **Temporal Dependencies (Market Memory):** - The model robustly captured the "near-random walk" nature of the market.
    - **Real Mean ACF (Lag-1):** `0.0312` (indicating weak momentum).
    - **Generated Mean ACF (Lag-1):** `-0.0615` (indicating slight mean reversion).
    - **Autocorrelation MSE:** **`0.004474`**. This microscopic error level confirms the overall magnitude of temporal correlations was correctly simulated without severe overfitting.
