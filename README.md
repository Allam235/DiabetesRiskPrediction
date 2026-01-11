# Hospital Readmission Prediction Neural Network

## Project Overview

This project implements a machine learning pipeline to predict **30-day hospital readmission risk** using structured electronic health record (EHR) data. The goal is to model complex, nonlinear relationships between clinical features while maintaining numerical stability and robustness on real-world, imbalanced medical data.

The core model is a **feedforward neural network implemented from scratch** (NumPy-based) with explicit handling of class imbalance, unstable gradients, and activation saturation—common issues in medical ML tasks.

---

## Problem Statement

Hospital readmissions within 30 days are costly and often preventable. Accurately identifying high-risk patients can support early intervention and resource allocation. This project frames readmission prediction as a **multiclass classification problem** using patient demographics, diagnoses, lab results, and treatment indicators.

---

## Dataset

* **Source:** Public clinical readmission dataset (100,000+ patient encounters)
* **Features:** 49 total features after preprocessing

  * Demographics (age group, race, gender)
  * Clinical measurements (glucose, A1C)
  * Diagnoses and specialties
  * Medication and treatment indicators
* **Target:** Readmission outcome within 30 days (3 classes)
* **Challenges:**

  * Severe class imbalance
  * High-cardinality categorical variables
  * Missing and corrupted values

---

## Feature Engineering

* **One Hot Encoding** for categorical clinical variables (race, age buckets, specialty, medications)
* **Mixed Feature Types:** Combination of categorical and continuous inputs
* **Input Normalization:** Gentle scaling using `(x - mean) / (3 * std)` to prevent activation instability
* **Missing Value Handling:** Default fallbacks to prevent NaNs or training crashes
* **Data Shuffling:** Applied prior to train/test split to remove ordering bias
* **Train/Test Split:** 2/3 training, 1/3 testing

---

## Model Architecture

* **Type:** Feedforward Neural Network (from scratch)
* **Layers:**

  * Dense: 64 units → Leaky ReLU → Batch Normalization
  * Dense: 32 units → Leaky ReLU → Batch Normalization
  * Output: 3 units → Softmax
* **Initialization:** Leaky-ReLU-aware He initialization
* **Activation Function:** Leaky ReLU (prevents dead neurons)
* **Output:** Softmax probabilities for multiclass prediction

---

## Training Strategy

* **Loss Function:** Weighted categorical cross-entropy
* **Class Imbalance Handling:** Inverse-frequency class weights
* **Optimizer:** Adam (implemented manually)
* **Learning Rate Schedule:** Inverse time decay
* **Mini-Batch Training:** Batch size = 256
* **Gradient Clipping:** Global norm capped at 1.0
* **Numerical Safety:**

  * Logit and probability clipping
  * Stable softmax implementation

---

## Evaluation & Debugging

* **Accuracy Evaluation:** Sample-based testing for efficiency
* **Dead Neuron Detection:** Variance-based activation monitoring
* **Debug Logging:**

  * Layer activations
  * Gradient norms
  * Weight updates
  * Loss trends

This instrumentation was used to identify and correct instability such as exploding gradients and neuron saturation.

---

## Key Learning Outcomes

* Implemented a full neural network training pipeline **without high-level ML frameworks**
* Gained hands-on experience with:

  * Batch normalization forward/backward passes
  * Gradient stability techniques
  * Imbalanced medical classification
* Learned how small preprocessing and scaling choices impact network behavior

---

## Limitations & Future Work

* Add calibration metrics (ECE, reliability diagrams)
* Visualize decision surfaces for selected clinical feature pairs
* Compare against baseline models (logistic regression, random forest)
* Extend to temporal models using sequential patient visits

---

## Technologies Used

* Python
* NumPy
