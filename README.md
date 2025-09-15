Name: Sayantika Chakraborty
Roll No.: ME22B190

# Assignment 4 – DA5401  
**Gaussian Mixture Models (GMM) for Synthetic Oversampling and Clustering-Based Undersampling (CBU)** 

## Overview  
This assignment builds upon Assignment 3, where we explored traditional resampling methods (SMOTE, CBO, CBU). In Assignment 4, the focus shifts to **probabilistic resampling using Gaussian Mixture Models (GMM)** and combining it with **Clustering-Based Undersampling (CBU)** for imbalanced classification (fraud detection). 

The primary objective is to generate **realistic synthetic samples** of the minority class (fraud) using GMM and evaluate how this affects classifier performance. 

## Contents  
1. **Part A: Data Preparation** 
   - Train-test split of the dataset. 
   - Isolation of minority (fraud) and majority (non-fraud) samples. 

2. **Part B: GMM for Minority Class** 
   - Fit a Gaussian Mixture Model (GMM) to the minority class. 
   - Determine the optimal number of components (`k`) using **AIC/BIC criteria**. 
   - Generate synthetic fraud samples by sampling from the fitted GMM. 
   - Combine these synthetic samples with the original dataset. 

3. **Part B.4: Rebalancing with CBU + GMM** 
   - Apply **Clustering-Based Undersampling (CBU)** to reduce the majority class. 
   - Combine with **GMM-generated synthetic fraud samples** to create a balanced dataset. 

4. **Part C: Performance Evaluation & Conclusion** 
   - Train Logistic Regression classifiers on: 
     - Baseline (Imbalanced) 
     - GMM Balanced 
     - CBU + GMM Balanced 
   - Evaluate using **Precision, Recall, F1-score** (Fraud class focus). 
   - Threshold tuning performed to optimize F1-score. 
   - Comparative analysis with bar charts. 
   - Final recommendation on the effectiveness of GMM for resampling. 

## Methods & Techniques  
- **Gaussian Mixture Model (GMM):** 
  Used to model the distribution of minority class data and generate new synthetic points. 

- **Threshold Tuning:** 
  Essential for GMM and resampled datasets. Default threshold (0.5) led to very poor precision/F1 due to oversensitivity to frauds. Optimizing threshold significantly improved results. 

- **Clustering-Based Undersampling (CBU):** 
  Applied to reduce the dominance of majority samples while preserving diversity using K-Means clustering. 

## Results (Summary)  
| Model               | Precision (Fraud) | Recall (Fraud) | F1-score (Fraud) |
|---------------------|-------------------|----------------|------------------|
| Baseline (Untuned)   | ~0.83             | ~0.64          | ~0.72            |
| GMM Balanced (Tuned) | ~0.62             | ~0.83          | ~0.72            |
| CBU + GMM (Tuned)    | ~0.81             | ~0.80          | ~0.81            |

- **Baseline:** High precision, moderate recall → misses many fraud cases. 
- **GMM Balanced:** Improved recall significantly, with moderate precision → better overall balance and competitive F1. 
- **CBU + GMM:** High recall and much improved precision → best F1-score, practical given the threshold tuning and increased majority scaling.

## Key Observations  
- **GMM oversampling** drastically improves recall, sometimes at large cost to precision, but with tuning and adjusted majority scaling it achieves much better precision and F1. 
- **Threshold tuning** is critical — it substantially improves precision without sacrificing recall, resulting in competitive F1 scores. 
- **Baseline model** remains strong for very high precision needs but is outperformed on balanced metrics by tuned GMM methods. 
- **CBU + GMM** benefits strongly from scaling the majority class to 100x (or larger) minority, leading to more realistic class proportions and better decision boundaries.

## Final Recommendation  
- **GMM-based oversampling with AIC/BIC and rebalanced CBU** is promising when combined with threshold tuning, yielding the best balanced fraud detection models. 
- The company should consider adopting these approaches for improved recall and F1, while maintaining acceptable precision. 
- Regular monitoring and cost-sensitive threshold adjustment remain essential. 
- The **baseline model** remains an option if minimizing false positives is paramount.

## Requirements  
- Python 3.x 
- Jupyter Notebook 
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn` 

