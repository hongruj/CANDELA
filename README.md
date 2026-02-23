# CANDELA: Cross-context Adaptive Neural Dynamics Embedding via Latent Array

PyTorch implementation of **CANDELA**, a framework for learning consistent and interpretable neural representations of motor cortex activity across **sessions, subjects, and tasks**.

## Overview
Brain-Computer Interfaces (BCIs) often struggle with context-dependent shifts in neural data distributions. CANDELA adaptively normalizes these shifts and extracts preserved latent dynamics using a fixed, learnable latent array. 

By fine-tuning only the input-output layers, CANDELA enables rapid, few-shot adaptation to unseen recordings and allows a pre-trained decoder to be directly reused. In our paper, we demonstrate its scalability by integrating it into the control loop of a **humanoid robot**.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/hongruj/CANDELA.git
cd CANDELA
pip install -r requirements.txt
```

## Repository Structure
```data/```: Scripts for downloading and preprocessing the electrophysiological datasets.

```src/```: Source code for the CANDELA model, training, and decoding.

```configs/```: Config files.

```experiments/```: Scripts to reproduce the main figures in the paper.

```robot_control/```: Deployment scripts for the humanoid robot (Unitree, G1).

## Quick Start
```bash
# Train on source context (e.g., Monkey J)
python run_hyperopt.py mj --samples 20
# Decode on source context (e.g., Monkey J)
python train.py --task behav --dataset_name mj
# Few-shot adaptation to target context (e.g., Monkey J to Monkey C)
python train.py --task new --dataset_name mj
```
