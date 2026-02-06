# Normalizing Flows & CycleGAN

## 📌 Overview

This repository contains implementations of two advanced deep generative model architectures: **Normalizing Flows** for anomaly detection and **CycleGAN** for unpaired image-to-image translation.

The project demonstrates how to:

1. **Detect Anomalies:** Using Masked Autoregressive Flows (MAF) to identify defects in industrial components by learning exact density estimation.
2. **Transfer Styles:** Using CycleGAN to translate images between domains (e.g., Summer  Winter) without paired training data.

---

## 🔍 Part 1: Anomaly Detection with Normalizing Flows

### Method: Masked Autoregressive Flow (MAF)

We implement a **Masked Autoregressive Flow (MAF)**, which transforms a simple base distribution (Gaussian) into the complex distribution of the data using a sequence of invertible transformations.

* **Core Architecture:** Masked Autoencoder for Distribution Estimation (MADE).
* **Objective:** Learn the log-likelihood  of "normal" data.

### Results: MVTec AD Dataset

The model was trained on the **Capsule** class of the MVTec AD dataset. By learning the density of defect-free capsules, the model assigns low likelihood scores to anomalies (scratches, cracks).

![maf](assets/MAF-convergence.png)
![maf](assets/distribution-anomaly-maf.png)

The model successfully separates normal samples (high likelihood) from anomalous defects (low likelihood tails).

---

## 🎨 Part 2: Unpaired Translation with CycleGAN

### Method: Cycle Consistency

CycleGAN learns mappings between two domains  and  by enforcing **Cycle Consistency**: .

* **Generator:** ResNet-based architecture (6 residual blocks).
* **Discriminator:** PatchGAN (70x70 receptive field).
* **Losses:** Adversarial Loss + Cycle Consistency Loss () + Identity Loss ().


### Results: Apple <-> Orange

Object transfiguration demonstrating texture and color mapping between fruit domains.

![maf](assets/apple2orange-epoch1-generated-samples.png)
![maf](assets/apple2orange-epoch20-generated-samples.png)
![maf](assets/apple2orange-loss-diagrams.png)
