# Adam-HD: An Adam Optimizer with Hierarchical and Dynamic Hyperparameters

## Project Overview

This repository contains the Python and PyTorch implementation of **Adam-HD**, an adaptive optimizer designed to improve upon the standard Adam algorithm. The core innovation of Adam-HD is a hierarchical control system that provides coordinated, real-time scheduling for its key hyperparameters: the learning rate ($\alpha$), momentum ($\beta$), and the adaptive memory coefficient ($\gamma$).

Unlike conventional methods that often rely on fixed hyperparameters or predefined learning rate schedules, Adam-HD leverages geometry-aware feedback from the optimization process to make more intelligent, holistic adjustments.

## Core Concepts & Methodology

The optimizer dynamically transitions between "aggressive" and "cautious" states by monitoring signals extracted directly from the training dynamics. The primary control signals are:

1.  **Relative Gradient Norm:** The L2 norm of the gradient at each step is compared against its own exponential moving average. This signal measures the "steepness" of the loss landscape relative to its recent history and is used to compute an `aggressiveness factor`.
2.  **Directional Consistency:** The cosine similarity between consecutive gradients is used to measure the smoothness and predictability of the optimization path.

A key component of the control system is a **Generalized Softsign function**, defined as $s(z, T) = z / (T + |z|)$. This function, controlled by a sensitivity gain (`k_agressividade`) and a flattening factor (`T`), maps the relative gradient norm to the aggressiveness factor without the premature saturation that affects standard sign functions.

This aggressiveness factor then directly modulates the learning rate ($\alpha$) and works in coordination with the directional consistency signal to adjust momentum ($\beta$) and adaptive memory ($\gamma$). This allows the optimizer to perform nuanced behaviors, such as intelligently reducing inertia when approaching a flat minimum to avoid overshooting.

## Repository Contents

This script is a self-contained implementation and analysis of the Adam-HD optimizer. It includes:

* The PyTorch implementation of the `AdamHDOptimizer` class.
* An experiment on a 2D synthetic loss function to provide a clear, qualitative visualization of the optimizer's smooth trajectory.
* A complete training and evaluation pipeline for a deep Multi-Layer Perceptron (MLP) on the **MNIST dataset**, which successfully converges and achieves high final accuracy (>98%).
* A suite of advanced analysis tools, including:
    * Time-series plots for all dynamic hyperparameters ($\alpha, \beta, \gamma$) and the gradient norm.
    * A numerical check and visualization for the Robbins-Monro convergence conditions.
    * A "Loss vs. Learning Rate" scatter plot derived from the main training data.

## How to Run

1.  Ensure you have the required libraries installed: `numpy`, `matplotlib`, `torch`, `torchvision`, and `scikit-learn`.
2.  Key parameters for the experiments, such as `K_AGRESSIVIDADE_GLOBAL`, `FLATTENING_FACTOR_T_GLOBAL`, and the number of epochs (`NUMERO_DE_EPOCAS`), can be adjusted in the global variables section at the top of the script.
3.  Run the Python script. It will first execute the 2D visualization and then proceed to the MLP training, followed by all the analysis plots.
