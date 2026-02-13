# Breast Cancer (Wisconsin) — ML Assignment 2

## a. Problem statement
Predict whether a breast mass is **malignant (1)** or **benign (0)** from diagnostic features using supervised learning. Build an end‑to‑end workflow: data preparation → model training (six algorithms on the **same dataset**) → evaluation on required metrics → saving trained pipelines → **Streamlit** UI for CSV upload, model selection, metrics, and confusion matrix/classification report.

---

## b. Dataset description
- **Dataset name:** Breast Cancer (Wisconsin)
- **Source (Kaggle):** https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset
- **Instances (rows):** 569
- **Features (columns used):** 32
- **Target column:** `diagnosis` (mapped as `M → 1`, `B → 0`)
- **Preprocessing summary:**
  - Dropped non‑predictive ID‑style columns if present (e.g., `id`, `Unnamed: 0`).
  - **Numerical** features: median imputation + standardization.
  - **Categorical** features (if any): most‑frequent imputation + one‑hot encoding (dense output so **GaussianNB** works).
  - Train/test split: **stratified 80/20**, fixed random seed for reproducibility.

---

## c. Models used: **Comparison table with evaluation metrics**

All models are trained and evaluated on the **same processed split**.  
Reported metrics: **Accuracy, AUC, Precision, Recall, F1 Score, MCC**

| ML Model Name              | Accuracy |   AUC    | Precision |  Recall  |    F1    |    MCC    |
|----------------------------|---------:|---------:|----------:|---------:|---------:|----------:|
| Logistic Regression        | 0.964912 | 0.996032 | 0.975000  | 0.928571 | 0.951220 | 0.924518  |
| Decision Tree              | 0.929825 | 0.924603 | 0.904762  | 0.904762 | 0.904762 | 0.849206  |
| kNN                        | 0.947368 | 0.980489 | 0.973684  | 0.880952 | 0.925000 | 0.887244  |
| Naive Bayes (Gaussian)     | 0.921053 | 0.989087 | 0.923077  | 0.857143 | 0.888889 | 0.829162  |
| Random Forest (Ensemble)   | 0.973684 | 0.994378 | 1.000000  | 0.928571 | 0.962963 | 0.944155  |
| XGBoost (Ensemble)         | 0.973684 | 0.992725 | 1.000000  | 0.928571 | 0.962963 | 0.944155  |
```

---

## d. Observations on model performance

| ML Model Name              | Observation about model performance |
|----------------------------|--------------------------------------|
| Logistic Regression        | Achieved high AUC (0.996) with strong accuracy (96.49%) and balanced precision-recall. Performs very well due to near-linear separability of the dataset. Fast training and stable performance. |
| Decision Tree              | Lower accuracy (92.98%) and MCC (0.849) compared to ensemble methods. While interpretable, it shows higher variance and slightly weaker generalization. |
| kNN                        | Good performance (94.74% accuracy) but slightly lower recall (0.881). Sensitive to scaling and choice of k. Performs reasonably well on this dataset. |
| Naive Bayes (Gaussian)     | Fastest training with decent AUC (0.989), but lower recall (0.857) and overall accuracy (92.10%). Assumption of feature independence slightly limits performance. |
| Random Forest (Ensemble)   | Best overall performance with highest accuracy (97.37%) and MCC (0.944). Perfect precision (1.000) and strong F1 (0.963). Ensemble averaging reduces variance and improves robustness. |
| XGBoost (Ensemble)         | Matches Random Forest in accuracy (97.37%) with very high AUC (0.993). Strong precision and F1, with faster training time than RF. Excellent predictive performance. |
