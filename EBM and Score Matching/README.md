Here is the README for the final project. I have prioritized the **"Best Practice" EBM implementation** (with the replay buffer) as requested and structured the document to clearly distinguish between the Energy-Based and Score-Based approaches.

As before, I have defined the **Visuals & Results** section first so you know exactly which images to export to your `assets/` folder.

---

# Deep Generative Models: Energy-Based & Score-Based Generative Models

## 📌 Overview

This repository explores two classes of implicit generative models that move beyond standard GANs and VAEs: **Energy-Based Models (EBMs)** and **Score-Based Generative Models (SGMs)**.

The project is divided into two distinct implementations:

1. **Energy-Based Models (EBM):** Learning an unnormalized probability density function  using Langevin Dynamics and a **Replay Buffer** for stable training.
2. **Score-Based Models (NCSN):** Learning the gradient of the log-density function  (the "score") using Denoising Score Matching and Annealed Langevin Dynamics.

## 📂 Project Structure

* `EBM-BestPractice.ipynb`: **(Primary EBM)** The optimized implementation using a **Replay Buffer** to stabilize training and prevent mode collapse.
* `score-based-model.ipynb`: Implementation of **NCSN (Noise Conditional Score Network)** for both unconditional and conditional generation.
* `EBM.ipynb`: A baseline EBM implementation (without replay buffer) for comparison.
* `Persion-Report.pdf`: Detailed technical report covering Contrastive Divergence, DSM, and noise scale analysis.

---

## ⚡ Part 1: Energy-Based Models (EBM)

### Methodology

We train a neural network  to assign low energy values to realistic data (MNIST) and high energy values to noise.

* **Training:** Uses **Contrastive Divergence** with Stochastic Gradient Langevin Dynamics (SGLD) to sample negative examples from the current model distribution.
* **Key Technique (Replay Buffer):** To improve sample quality and convergence speed, we maintain a buffer of past generated samples. These are used as initialization points for the MCMC chain, preventing the model from forgetting previously learned modes.

### Tasks

* **Generation:** Generating digits from pure noise via iterative energy minimization.
* **Denoising:** restoring clean images from noisy inputs by following the gradient of the energy function .

---

## 🎼 Part 2: Score-Based Generative Models (SBM)

### Methodology

We implement a **Noise Conditional Score Network (NCSN)**. Instead of learning the density directly, we learn the *score function* (the direction to move towards high-density regions).

* **Score Matching:** Trained using **Denoising Score Matching (DSM)**, where the model learns to denoise samples perturbed by various levels of Gaussian noise ().
* **Sampling:** Uses **Annealed Langevin Dynamics**, starting with high noise levels (coarse structure) and gradually reducing noise (fine details).

### Conditional Generation

We extend the NCSN to perform **Class-Conditional Generation**  by injecting class labels into the score network, allowing us to control which digit is generated.

---

## 📊 Visuals & Results to Include

Please export the following figures from your report or notebooks and place them in the `assets/` folder.

### 1. EBM Generation (Evolution)

* *What to add:* The grid showing the transition from noise to digits over MCMC steps.
* **Filename:** `assets/ebm_generation_evolution.png`
* **Caption:** *Evolution of MCMC chains in the Energy-Based Model. Starting from random noise (left), samples converge to realistic MNIST digits (right) as energy is minimized.*

### 2. SBM Conditional Generation

* *What to add:* The grid where each row corresponds to a specific digit class (0-9).
* **Filename:** `assets/sbm_conditional_grid.png`
* **Caption:** *Conditional samples from the Score-Based Model. Each row corresponds to a specific class label (0-9), demonstrating the model's ability to control generation.*

### 3. Denoising Results (EBM or SBM)

* *What to add:* The comparison showing "Original | Noisy | Denoised".
* **Filename:** `assets/denoising_results.png`
* **Caption:** *Image denoising using Langevin Dynamics. The model successfully removes additive Gaussian noise to recover the original structure.*

### 4. Loss Curves (Comparison)

* *What to add:* The loss plot showing the stability of the Replay Buffer implementation vs. the baseline.
* **Filename:** `assets/training_stability.png`
* **Caption:** *Training stability analysis. The Replay Buffer (Orange) prevents the diverging loss oscillations seen in the baseline approach (Blue).*

---

## 💻 Usage

### 1. Energy-Based Model (Recommended)

Open `EBM-BestPractice.ipynb`. This notebook implements the Replay Buffer technique.

```python
# To sample from the trained EBM:
samples = replay_buffer.sample(batch_size)
final_images = langevin_dynamics(model, samples)

```

### 2. Score-Based Model

Open `score-based-model.ipynb` for NCSN training.

```python
# Conditional Sampling (e.g., generate digit '5')
labels = torch.tensor([5] * batch_size)
samples = annealed_langevin_dynamics(model, labels, sigmas)

```

## 📚 References

* *A Theory of Generative ConvNet (2016)* - Wu et al.
* *Generative Modeling by Estimating Gradients of the Data Distribution (2019)* - Song & Ermon.

---

*Author: Artin Tavasoli*