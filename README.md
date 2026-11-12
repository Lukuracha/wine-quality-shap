# Explainable Wine Quality Classification with SMOTE and SHAP: A Comparison of Red and White Wines

This repository contains the official source code and experimental pipeline for the paper: **"Explainable Wine Quality Classification with SMOTE and SHAP: A Comparison of Red and White Wines"**. 

## 1. Overview

This project investigates the prediction of wine quality using physicochemical properties. The task is formulated as a three-class imbalanced classification problem (Low, Medium, High). The study compares the predictive performance of Random Forest, XGBoost, and Support Vector Machine (SVM) under original and SMOTE-balanced training conditions. Furthermore, it employs Explainable Artificial Intelligence (XAI) techniques, specifically Kernel SHAP and TreeSHAP, to interpret feature contributions and compares these findings with the original sensitivity analysis reported by Cortez et al. (2009).

## 2. Dataset

The experiments utilize the Wine Quality datasets introduced by Cortez et al. (2009) and distributed through the UCI Machine Learning Repository. 
- Red wine dataset: 1,599 samples (1,359 after deduplication)
- White wine dataset: 4,898 samples (3,961 after deduplication)
- Features: 11 physicochemical properties (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol).

The code automatically downloads the datasets from the UCI repository upon execution.

## 3. Repository Structure

- `wine_quality_pipeline.ipynb`: The main Jupyter Notebook containing the complete experimental workflow. This notebook was developed and executed using Google Colab.
- `requirements.txt`: The Python package dependencies required to run the notebook.
- `README.md`: Documentation for the repository.

## 4. Running the Code (Google Colab)

The primary environment used for this study is Google Colab. To run the experiments, you can either:

**Option A: Open directly in Google Colab (Recommended)**
You can view the full source code, including the rendered outputs and charts, directly in your browser without downloading anything:
[Open `wine_quality_pipeline.ipynb` on GitHub](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/blob/main/wine_quality_pipeline.ipynb)
**Option B: Run Locally**
If you prefer to run the code locally, clone this repository, install the dependencies using `pip install -r requirements.txt`, and open the `.ipynb` file in Jupyter Notebook or JupyterLab.

## 5. Experimental Workflow

The Jupyter Notebook is structured to follow the methodology described in the paper:

1. **Data Preprocessing**: Removal of duplicate records and grouping quality scores into three ordinal categories: Low (3-4), Medium (5-6), and High (7-9).
2. **Data Splitting**: A stratified 80:20 train-test split (random state = 42).
3. **Scaling**: Min-Max scaling fitted exclusively on the training data and refitted within each cross-validation fold to prevent data leakage.
4. **Model Training**: 
   - Classification: Random Forest, XGBoost, and SVM (with and without SMOTE).
   - Regression: Linear Regression and Support Vector Regression (SVR) for continuous score prediction.
5. **Evaluation**: 5-fold stratified cross-validation and held-out test set evaluation using Accuracy, Macro-F1, and Weighted-F1. Per-class recall and precision are also generated.
6. **Explainability**: 
   - Kernel SHAP applied to the SVR models to rank feature importance.
   - TreeSHAP applied to the Random Forest + SMOTE models.
   - Spearman rank correlation computed to compare SHAP rankings with the sensitivity analysis of Cortez et al. (2009).

## 6. Environment and Requirements

To reproduce the results, the following main dependencies are required (detailed in `requirements.txt`):
- `scikit-learn==1.6.1`
- `xgboost==3.4.1`
- `imbalanced-learn==0.14.2`
- `shap==0.52.0`
- `pandas`, `numpy`, `matplotlib`, `scipy`

## 7. Reproducibility Note

All random states in the dataset splitting, model initialization, and SMOTE generation are fixed at `random_state = 42`. Because Kernel SHAP utilizes stochastic sampling to estimate Shapley values, the global random seed (`np.random.seed(42)`) is explicitly set prior to the SHAP calculation to ensure reproducibility. 

Please note that due to the floating-point non-associativity and the sampling nature of Kernel SHAP, slight variations in the ranking of lower-importance features may occur across different hardware configurations. The paper explicitly discusses this run-dependency and focuses interpretations on the top-ranked features.

## 8. Citation

If you use this code or find our findings useful in your research, please consider citing our paper (details will be updated upon publication):

```bibtex
@inproceedings{tran2026explainable,
  title={Explainable Wine Quality Classification with SMOTE and SHAP: A Comparison of Red and White Wines},
  author={Tran, Nhat and Pham, Danh and Nguyen, Dat and Le, Thu and Le, Nhat-Tung},
  booktitle={Lecture Notes in Computer Science (LNCS)},
  year={2026},
  organization={Springer}
}
