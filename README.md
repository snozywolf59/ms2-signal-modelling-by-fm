# Mass Spectrometry Signal Modeling with Flow Matching

This repository documents my research on applying Flow Matching to model tandem mass spectrometry (MS/MS) signals in proteomics.

The project explores generative modeling techniques for predicting fragment ion intensity distributions from peptide sequences and precursor charge states.

---

# Overview

Mass spectrometry-based proteomics relies heavily on accurate MS/MS spectrum prediction.
This repository investigates whether Flow Matching can effectively model the distribution of fragment ion intensities and capture complex fragmentation patterns.

The project includes:

- Flow Matching experiments on synthetic 2D toy datasets
- Neural architectures for peptide and charge embeddings
- MS/MS spectrum generation and reconstruction
- Training and evaluation pipelines on real proteomics datasets
- Experimental notebooks and visualization demos

---

# Dataset

This work uses the **ProteomeTools Fragmentation Intensity** dataset provided by ProteomicsML.

- Dataset: ProteomicsML ProteomeTools Fragmentation Intensity Dataset
- URL: https://proteomicsml.org/datasets/fragmentation/ProteomeTools_FI.html

Please download the dataset manually and configure the dataset path in the `.env` file.

Example:

```bash
TRAIN_PATH=data\traintest_hcd.hdf5
TEST_PATH=data\holdout_hcd.hdf5

```

See `.env_example` for reference.

---

# Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt -q
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q
```

---

# Training

Modify configuration if needed:

```bash
run_real_data/config.py
```

Train the model:

```bash
python run_real_data/train.py
```

Evaluate the model:

```bash
python run_real_data/eval.py
```

---

The example result in: run_real_data\train-test-hcd-001 (2).ipynb

# Repository Structure

```text
.
├── miniprojects/
│   └── Experiments for understanding Flow Matching on toy 2D problems
│
├── funny-demo-gif/
│   └── Visualization demos and Flow Matching animations
│
├── papers/
│   └── Related papers and reading materials
│
├── run_real_data/
│   ├── models/
│   │   └── Embedding modules and Flow Matching architectures
│   │
│   ├── utils/
│   │   └── Utility functions and preprocessing
│   │
│   ├── train.py
│   ├── eval.py
│   └── config.py
│
├── requirements.txt
├── .env_example
└── README.md
```

---

# Research Goals

The primary goals of this project are:

- Investigate Flow Matching for spectrum generation
- Model fragment ion intensity distributions
- Learn peptide representations for MS/MS prediction
- Explore continuous generative trajectories in spectral space

---

# Current Features

- Peptide sequence embedding
- Charge embedding
- Time-conditioned Flow Matching model
- Transformer-based architectures
- Support for masked ion prediction
- Evaluation with spectrum similarity metrics
- Visualization tools for trajectory dynamics

---

# Future Work

- Conditional generation for instrument settings
- Larger transformer backbones
- Diffusion/Flow hybrid models
- Improved decoding for sparse spectra
- Benchmark comparison with existing spectrum predictors
- Support for modified peptides

---

# References

If you find this repository useful, please also consider exploring related work in:

- Flow Matching
- Generative modeling
- Computational proteomics
- MS/MS spectrum prediction

Relevant papers are collected in the `papers/` directory.

---

# Disclaimer

This repository is primarily a personal research and learning project.
The codebase is experimental and may change frequently.
