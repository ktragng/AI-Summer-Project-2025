# AI Summer Project 2025: Investigating Grokking Acceleration in Deep Neural Networks

## Project Overview
This project explores the effects of **gradient filtering** on **grokking acceleration** in deep neural networks. The goal is to analyze how filtering impacts **learning dynamics, feature rank evolution, and generalization**.

## Research Questions
- How does **gradient filtering** affect **grokking acceleration**?
- What impact does it have on **feature rank evolution**?
- Can it **enhance generalization** in deep learning models?

## Methodology
- **Model:** Multi-Layer Perceptron (MLP)
- **Dataset:** Modular arithmetic dataset
- **Key Experiment:** Compare a **baseline model** vs. a **filtered model** using accuracy analysis and feature rank evolution.

## Project Structure

main.ipynb – Jupyter Notebook for running experiments and visualization
models/ – Jupyter Notebook for saving and loading trained models
train/ – Jupyter Notebook for training models
utils/ – Python utilities for data processing and training
__pycache__/ – Python cache files


## Saved Model Checkpoints

results_1500_epochs.pth – Model trained for 1500 epochs
results_500_epochs.pth – Model trained for 500 epochs
results_advanced.pth – Advanced training results
results.pth – General model results
mlp_model.pth – Trained MLP model
loss_history.pth – Training loss history


## Key Findings
- **Filtered models learn slower** but exhibit **different feature rank evolution**.
- Feature rank **stabilization** might **affect generalization**.
- **Gradient filtering modifies** how networks learn, requiring further study.

## Future Work
- Implement **adaptive filtering strategies**.
- Extend experiments to **other neural architectures**.
- Investigate the link between **rank stability and generalization**.

## Contact
📧 [Your Email]  
🔗 [Your LinkedIn (if applicable)]  

