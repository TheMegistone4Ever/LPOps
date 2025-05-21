# LPOps (Linear Programming Operations)

###### &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - by [Mykyta Kyselov (TheMegistone4Ever)](https://github.com/TheMegistone4Ever), based on research by A. Pavlov, M. Kyselov, V. Kobelskyi, A. Kushch.

LPOps is a Python project dedicated to the theoretical and empirical analysis of the computational complexity of the
standard simplex method for solving Linear Programming (LP) problems in canonical form. The primary focus is on
estimating the number of arithmetic operations performed by the algorithm. This project conducts an experimental study
of existing empirical formulas, develops and validates new empirical formulas, and compares their accuracy in predicting
the operational cost of the simplex method.

## Table of Contents

1. [Introduction](#1-introduction)
    1. [Problem Statement](#11-problem-statement)
    2. [Project Goals](#12-project-goals)
2. [Theoretical Background](#2-theoretical-background)
    1. [Simplex Method and LP Problems](#21-simplex-method-and-lp-problems)
    2. [Complexity Models](#22-complexity-models)
        1. [Borgwardt's Model](#221-borgwardts-model)
        2. [Smoothed Analysis Model](#222-smoothed-analysis-model)
        3. [Adler-Megiddo Model](#223-adler-megiddo-model)
        4. [Refined Borgwardt Model](#224-refined-borgwardt-model)
        5. [General Polynomial Model](#225-general-polynomial-model)
        6. [General Polynomial Model with Logarithmic Term](#226-general-polynomial-model-with-logarithmic-term)
        7. [General Mixed Model](#227-general-mixed-model)
        8. [Weighted Ensemble Model](#228-weighted-ensemble-model)
3. [Experimental Methodology](#3-experimental-methodology)
    1. [LP Problem Generation](#31-lp-problem-generation)
    2. [Simplex Method Implementation & FLOP Counting](#32-simplex-method-implementation--flop-counting)
    3. [Data Sampling and Caching](#33-data-sampling-and-caching)
    4. [Model Fitting and Evaluation](#34-model-fitting-and-evaluation)
4. [Key Findings and Results](#4-key-findings-and-results)
    1. [Empirical Formula from Research](#41-empirical-formula-from-research)
    2. [Results from `analysis.py`](#42-results-from-analysispy)
5. [Visualization of Empirical Analysis](#5-visualization-of-empirical-analysis)
6. [Prerequisites](#6-prerequisites)
    1. [System Requirements](#61-system-requirements)
    2. [Software Requirements](#62-software-requirements)
7. [Installation & Setup](#7-installation--setup)
    1. [Clone Repository](#71-clone-repository)
    2. [Setup Virtual Environment (Recommended)](#72-setup-virtual-environment-recommended)
    3. [Install Dependencies](#73-install-dependencies)
8. [Running the Analysis](#8-running-the-analysis)
9. [Project Structure](#9-project-structure)
10. [Code Overview](#10-code-overview)
11. [License](#11-license)

## 1. Introduction

### 1.1 Problem Statement

The simplex method is a fundamental algorithm for solving Linear Programming (LP) problems. While its worst-case
complexity can be exponential, its practical performance is often polynomial. Understanding and predicting the
average-case complexity, specifically the number of arithmetic operations (Floating Point Operations - FLOPs), is
crucial for various applications, including algorithm design, performance optimization, and resource estimation for
large-scale problems.

This project addresses the challenge of accurately estimating the number of arithmetic operations performed by the
standard simplex method when solving LP problems of varying sizes (number of constraints $m$ and variables $n$).

### 1.2 Project Goals

The main goals of this project are:

* To conduct an empirical analysis of the standard simplex method's operational complexity.
* To evaluate existing theoretical and empirical formulas for estimating the number of arithmetic operations.
* To develop and validate a new, more accurate empirical formula based on statistical simulation and model fitting
  techniques (inspired by GMDH - Group Method of Data Handling).
* To provide a framework for generating LP problems, running the simplex method, collecting performance data, fitting
  complexity models, and visualizing results.
* To identify the model that provides the best accuracy in predicting the simplex method's operational cost based on
  problem dimensions.

## 2. Theoretical Background

### 2.1 Simplex Method and LP Problems

The project focuses on LP problems in canonical form. Typically, this means problems of the form:
Minimize $z = c^T x$
Subject to $Ax = b$, $x \ge 0$.

The `generator.py` script in this project generates LP problems where an initial feasible solution is known, with
constraints $A'x' \le b'$. The `simplex.py` script then converts these to the equality form $Ax = b$ by adding slack
variables, where $A = [A' | I]$ and $x = [x'^T | x_{slack}^T]^T$. The number of constraints remains $m$, and the number
of variables becomes $n' + m$. In the context of the complexity models, $n$ usually refers to the number of variables in
the final tableau ($n' + m$ after adding slacks). However, `analysis.py` uses $n$ as the original number of
variables $n'$. The models are defined with $m$ and $n$ (original variables) as inputs.

### 2.2 Complexity Models

Several models are investigated to describe the relationship between problem dimensions ($m, n$) and the number of
operations. $X$ represents the input tuple $(m, n)$.

#### 2.2.1 Borgwardt's Model

Based on Borgwardt's analysis of the average number of pivot steps, a common interpretation for total operations is:

$$
\text{Ops}(m,n) = a \cdot m^3 \cdot n
$$

* `borgwardt_model(X, a)`

#### 2.2.2 Smoothed Analysis Model

Derived from Spielman and Teng's smoothed analysis:

$$ \text{Ops}(m,n) = a \cdot m \cdot n^5 \cdot (\ln n)^b $$

* `smoothed_analysis_model(X, a, b)`

#### 2.2.3 Adler-Megiddo Model

Suggests a polynomial bound, often simplified for operations as:
$$ \text{Ops}(m,n) = a \cdot n^4 $$

* `adler_megiddo_model(X, a)`

#### 2.2.4 Refined Borgwardt Model

A variation or refinement of Borgwardt's model:
$$ \text{Ops}(m,n) = a \cdot m^3 \cdot n^2 $$

* `refined_borgwardt_model(X, a)`

#### 2.2.5 General Polynomial Model

A flexible polynomial form:
$$ \text{Ops}(m,n) = a \cdot m^b \cdot n^c $$

* `general_model(X, a, b, c)`

#### 2.2.6 General Polynomial Model with Logarithmic Term

Extends the general model to include a logarithmic dependency on $n$:
$$ \text{Ops}(m,n) = a \cdot m^b \cdot n^c \cdot (\ln n)^d $$

* `general_model_log(X, a, b, c, d)`

#### 2.2.7 General Mixed Model

A more complex model combining two polynomial-logarithmic terms, found to be effective in capturing observed
complexities:
$$ \text{Ops}(m,n) = a \cdot m^b \cdot n^c \cdot (\ln n)^d + e \cdot m^f \cdot n^g $$

* `general_model_mixed(X, a, b, c, d, e, f, g)`

#### 2.2.8 Weighted Ensemble Model

Combines predictions from multiple base models using learned weights:
$$ \text{Ops}(m,n) = \sum_{k} (\text{weight}_k \cdot \text{model}_k(X, \text{params}_k)) $$

* `weighted_ensemble_model(X, list_of_model_tuples)`
  where each tuple is `(model_function, weight, model_parameters)`.

## 3. Experimental Methodology

### 3.1 LP Problem Generation

LP problems are generated using `generator.py:generate_lp_problem(m, n, factor)`.
This function creates:

* $c$: Objective function coefficients (size $n$).
* $A$: Constraint matrix (size $m \times n$).
* $b$: Right-hand side values (size $m$).
* $x_{true}$: A known feasible solution.
  The elements of $A$, $c$, and the initial $x_{true}$ are drawn from random distributions, scaled by `factor`. $b$ is
  calculated as $Ax_{true}$ plus some slack to ensure feasibility.

The analysis script `analysis.py` typically uses:

* `m_range = list(range(200, 2001, 50))`
* `n_range = list(range(200, 2001, 50))`
* `num_runs = 5` (number of problems for each $m, n$ pair)

### 3.2 Simplex Method Implementation & FLOP Counting

The `simplex.py:simplex_method(c, A, b)` function implements the standard simplex algorithm (two-phase not explicitly
mentioned, assumes initial BFS from slack variables). Key features:

* **Slack Variables**: Converts $Ax \le b$ to $Ax' = b$ by adding slack variables.
* **Operation Counting**: Meticulously counts FLOPs (multiplications, divisions, additions, subtractions are generally
  counted) for each major step:
    * Calculating reduced costs.
    * Checking for optimality.
    * Selecting entering variable (argmax).
    * Minimum ratio test (argmin).
    * Updating the basis and tableau.
* **Large Matrix Handling**:
    * Use block processing for some matrix operations to manage memory.
    * Switches between direct matrix inversion (`numpy.linalg.inv`) and LU factorization (`scipy.linalg.lu_factor`,
      `lu_solve`) based on matrix size (`inversion_threshold`) for solving linear systems involving the basis
      matrix $B$.

### 3.3 Data Sampling and Caching

* **Data Generation**: The `analysis.py` script generates multiple datasets (samples W1, W2, L1, L2) by varying the
  `factor` in `generate_lp_problem`.
* **Caching**: `cache_manager.py:CacheManager` is used extensively to store and retrieve:
    * Generated sample data (tuples of $m, n, \text{operation_count}$).
    * Results of individual simplex method runs.
    * Fitted model parameters.
    * Data extracted for plotting.
      This avoids re-computation during iterative analysis or multiple runs.

### 3.4 Model Fitting and Evaluation

* **Loss Function**: The `utils.py:calculate_loss` function computes the Mean Squared Error (MSE) between the predicted
  operation counts from a model and the actual counts observed from simplex runs.
* **Parameter Optimization**: `scipy.optimize.minimize` (with the "Nelder-Mead" method) is used in
  `analysis.py:fit_model` to find the optimal parameters for each complexity model by minimizing the MSE loss.
* **Model Comparison**:
    * Models are trained on one sample (e.g., W1) and parameters are recorded.
    * These parameters can be used as initial guesses for fitting on another sample (e.g., W2) or parameters from W1 and
      W2 fits are averaged.
    * Models are evaluated based on their MSE scores on various datasets.
    * A weighted ensemble model is also constructed and evaluated.
* **Plotting**: `utils.py:create_plots` generates 2D and 3D visualizations of the data and model fits, comparing actual
  vs. predicted operations across ranges of $m$ and $n$. These are saved in the `plots/` directory.

## 4. Key Findings and Results

### 4.1 Empirical Formula from Research

The associated research paper (by Pavlov, Kyselov, et al.) highlights that the **General Mixed Model** (Section 2.2.7)
provides a particularly good fit for the empirical data. The derived formula for the number of operations ($N_{ops}$)
is:

$$
N_{ops}(m,n) \approx 0.63 \cdot m^{2.96} \cdot n^{0.02} \cdot (\ln n)^{1.62} + 4.04 \cdot m^{-4.11} \cdot n^{2.92}
$$

This formula was identified as providing the best accuracy (minimum Sum of Squared Errors (SSE)) among the models
considered in that study.

### 4.2 Results from `analysis.py`

Running the `analysis.py` script will produce detailed logs in `analysis.log` and to the console, including:

* **Parameters for each model**: Fitted coefficients for all models listed in Section 2.2 on different datasets (W1,
  W2).
* **Scores (MSE)**: Performance of each model on these datasets.
* **Averaged parameters and scores**: Stability of parameters across datasets.
* **Weighted Ensemble Model**: Constitution and performance of the ensemble model on datasets L1 and L2.
* **Summary**: Identification of the best single model and a comparison with the weighted ensemble model based on
  average scores.

The script is designed to confirm or refine such findings and provide a robust comparison across different model
structures. Should consult the output of `analysis.py` for the specific coefficients and scores determined by the
current run.

## 5. Visualization of Empirical Analysis

The `analysis.py` script generates various plots to help visualize the relationship between problem size ($m, n$),
actual operations, and the fitted models. These are saved in the `plots/` directory. Below are example visualizations
for the "General Mixed" model on the W1 dataset (logarithmic scale), similar to those presented in the associated
research.

*2D Plot - General Mixed Model (Log Scale) - W1 Dataset (Operations vs. m, Operations vs. n):*

<img src="images/General Mixed_W1_log_2d.png" alt="2D Plot - General Mixed (Log Scale) - W1" width="600"/>

*3D Plot - General Mixed Model (Log Scale) - W1 Dataset:*

<img src="images/General Mixed_W1_log_3d.png" alt="3D Plot - General Mixed (Log Scale) - W1" width="600"/>

## 6. Prerequisites

### 6.1 System Requirements

* **Operating System:** Windows, macOS, or Linux.
* **CPU:** A modern multicore processor is beneficial but not strictly required.
* **RAM:** 4GB+; 8GB or more is recommended for analyzing larger problem instances or generating extensive datasets, due
  to data caching and processing.

### 6.2 Software Requirements

* **Python:** Version 3.9 or newer (developed and tested with Python 3.9+).
* **Pip:** For installing Python packages.
* **Dependencies:** As listed in `requirements.txt`:
    * `numpy~=2.2.2`
    * `matplotlib~=3.10.0`
    * `pandas~=2.2.3`
    * `scipy~=1.15.1`

## 7. Installation & Setup

### 7.1 Clone Repository

```bash
git clone https://github.com/TheMegistone4Ever/LPOps.git
cd LPOps
```

### 7.2 Setup Virtual Environment (Recommended)

Using a virtual environment is highly recommended to manage dependencies and avoid conflicts with other Python projects.

```bash
# Create a virtual environment (e.g., named .venv)
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 7.3 Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

## 8. Running the Analysis

To run the full analysis pipeline:

```bash
python analysis.py
```

**Outputs:**

* **Console Log:** Progress, fitted parameters, scores, and summary results will be printed to the console.
* **Log File:** Detailed logs are saved to `analysis.log`.
* **Cache Files:** Intermediate results (samples, simplex solutions, model fits) are saved in the `cache/` directory.
  This can speed up later runs if parts of the analysis have already been completed.
* **Plots:** Visualization plots are saved in the `plots/` directory.

To clear previously cached results and start fresh, delete the contents of the `cache/` directory.

## 9. Project Structure

```
LPOps/
├── .venv/                          # Virtual environment (if created)
├── cache/                          # Directory for cached data and results
├── images/                         # Images used in README.md
│   ├── General Mixed_W1_log_2d.png
│   └── General Mixed_W1_log_3d.png
├── plots/                          # Output directory for generated plots
├── .gitattributes                  # Git attributes for line endings and diffs
├── .gitignore                      # Specifies intentionally untracked files
├── analysis.log                    # Log file from the analysis script
├── analysis.py                     # Main script to run the complexity analysis
├── cache_manager.py                # Manages saving and loading of cached objects
├── generator.py                    # Generates random LP problems
├── LICENSE.md                      # Project license file (Assumed, ensure it exists)
├── models.py                       # Defines the mathematical complexity models
├── README.md                       # This file
├── requirements.txt                # Python package dependencies
├── simplex.py                      # Implementation of the simplex method with FLOP counting
└── utils.py                        # Utility functions (loss calculation, plotting helpers)
```

## 10. Code Overview

* **`analysis.py`**: Orchestrates the entire analysis. It generates data samples, fits various complexity models to this
  data, evaluates their performance, and logs/prints the results. It also manages the creation of plots.
* **`simplex.py`**: Contains the core implementation of the simplex method. Crucially, it includes logic to count the
  number of floating-point operations (FLOPs) performed during the solution process.
* **`models.py`**: Defines the mathematical functions for the different complexity models being tested (e.g., Borgwardt,
  General Mixed, etc.).
* **`generator.py`**: Responsible for creating random LP problem instances ($c, A, b$) with specified
  dimensions ($m, n$) and a scaling factor.
* **`cache_manager.py`**: Implements a caching system to store and retrieve intermediate results (like generated LP
  data, simplex solutions, fitted model parameters). This is vital for efficiency, especially when dealing with
  time-consuming computations or large datasets.
* **`utils.py`**: Provides helper functions used across the project, including the loss calculation (Mean Squared
  Error), data extraction for plots, and the plot generation logic using Matplotlib.
* **`requirements.txt`**: Lists all external Python libraries required for the project to run.
* **`.gitattributes` & `.gitignore`**: Standard Git configuration files.

## 11. License

The project is licensed under the [CC BY-NC 4.0 License](LICENSE.md).