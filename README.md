# 🧠 In-Context Learning on Periodic Functions

### Exploring the Limits of Modern Architectures on Sinusoidal Function Learning

---

## 📘 Overview

This project investigates whether modern neural architectures — including **Transformers**, **Mamba**, **RNNs**, and **Fourier-based MLPs** — possess the intrinsic ability to perform **in-context learning (ICL)** on **periodic functions** such as

\[
f(x) = \sin(Wx + b)
\]

without explicit gradient updates or labeled supervision.

While large language models have shown strong ICL behavior on linear and simple nonlinear functions, their ability to infer latent **periodic structures** remains poorly understood. This project aims to uncover whether these architectures can *implicitly learn the functional rule* from contextual examples

\[
(x_1, f(x_1)), (x_2, f(x_2)), \ldots, (x_n, f(x_n))
\]

and generalize to unseen points.

---

## 🧩 Motivation

Periodic functions such as sinusoids exhibit **circular topology**, which is fundamentally different from the **Euclidean geometry** underlying standard neural architectures. We explore how this mismatch affects in-context generalization and whether architectural modifications — such as the **Fourier MLP** or **CausalConv1D-enhanced Mamba** — can bridge this gap.

---

## ⚙️ Architectures Evaluated

- 🧱 **Transformer (GPT-2)** – A baseline architecture to test self-attention’s ability to capture periodic dependencies.  
- 🐍 **Mamba-1.4B** – A selective state-space model with potential long-range context handling.  
- 🔁 **RNN** – A recurrent model serving as a sequential baseline with limited memory.  
- 🔢 **Fourier-MLP** – A frequency-domain variant that explicitly encodes sinusoidal priors.

---

## 📊 Key Findings

- **Transformers & Mamba** demonstrate *high instability* and *large prediction variance* across random seeds.  
- **RNNs** perform adequately in low-variance regimes but suffer from **memorization** instead of true function abstraction.  
- **Fourier-MLP** achieves strong results in **low-dimensional** settings but fails to scale effectively as dimensionality grows.  
- Theoretically, we show that:
  - The **mutual information** between inputs and outputs for sinusoidal mappings is significantly lower than for linear ones.  
  - This induces a **sample complexity of O(d²)** in d-dimensional spaces.  
  - The **Euclidean nature** of most neural architectures is **incompatible with circular topologies**, explaining the observed failures.

---

## 🧠 Theoretical Insight

We propose that the **topological mismatch** between Euclidean representation spaces and circular function manifolds imposes an **information-theoretic bottleneck** on in-context generalization. This insight may guide future work on **topology-aware neural architectures** or **non-Euclidean representations** for learning periodic phenomena.

---

## 🧪 Experimental Setup

- Implemented in **PyTorch**
- Curriculum learning strategy gradually increases:
  - Input dimensionality  
  - Sinusoid frequency range  
  - Context length  
- Evaluation metrics: MSE, stability index, and mutual information estimates

---

## 🔍 Results Summary (example)

| Model | Dimensionality | Avg. MSE | Stability Index | Notes |
|-------|----------------|----------:|----------------:|-------|
| Transformer (GPT-2) | low | 0.12 | low | High variance across seeds |
| Mamba-1.4B | low | 0.18 | low | Divergent in high-frequency regimes |
| RNN (GRU) | low | 0.08 | med | Good but tends to memorize |
| Fourier-MLP | low | 0.04 | high | Best in low-d settings |
| Fourier-MLP | high | 0.35 | low | Fails to scale |

> Replace the numbers above with your empirical results / CSV outputs.

---

## 📂 Repository Structure

