# Optical Flow Implementation - VISION (Master IMA/DIGIT)

This repository contains the implementation of classical and advanced optical flow estimation methods as part of the VISION course at Sorbonne Université. The project focuses on differential methods for estimating motion between consecutive image frames.

## Project Overview

The objective is to implement and compare several optical flow techniques, ranging from global variational methods to local window-based approaches and multiresolution frameworks.

## Implemented Methods

### 1. Classical Differential Methods (PW1)

Horn-Schunck (Global): An iterative method that minimizes a functional combining the Optical Flow Constraint (OFC) with a global smoothness term.

Lucas-Kanade (Local): A local approach that solves the OFC by assuming a constant velocity within a small spatial window ($n \times n$).

### 2. Advanced Methods (PW2)

Nagel Method: An oriented regularization approach that incorporates second-order spatial derivatives ($I_{xx}$, $I_{yy}$, $I_{xy}$) to prevent smoothing across intensity boundaries.

Automatic Differentiation & Optimization: Using PyTorch and the LBFGS optimizer to minimize the Horn-Schunck cost function. This implementation supports:

$L_2$ norm (Standard Horn-Schunck)

$L_1$ and Huber norms for robust estimation against outliers.

Pyramidal Lucas-Kanade: A multiresolution framework using Gaussian pyramids to handle large displacements that violate the small-motion assumption of standard differential methods.

## Evaluation Metrics

When ground truth data is provided, the performance of the estimated flow field ($w_e$) is compared against the reference ($w_r$) using:

- End Point Error (EPE): $||w_r - w_e||$

- Angular Error (Space-Time):


$$\theta(i,j) = \arccos\left(\frac{1 + w_r \cdot w_e}{\sqrt{1 + ||w_r||^2}\sqrt{1 + ||w_e||^2}}\right)$$

- Norm Error: $||w_r|| - ||w_e||$

- Relative Norm Error: $\frac{||w_r|| - ||w_e||}{||w_r||}$

## Requirements

Python 3.x

NumPy: For matrix operations and numerical solvers.

Pillow (PIL): For image loading.

Matplotlib: For visualization (quiver plots and color maps).

PyTorch: Required for the automatic differentiation and LBFGS optimization tasks.

Scikit-Image: Required for warp and resize functions in the multiresolution implementation.


Visualization is performed using flowToColor() (from the provided middlebury module) or quiver() to display vector fields.
