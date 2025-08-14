# Survival Outcome Prediction in Breast Cancer (BRCA)

## Project Overview

This project applies **machine learning** to predict **breast cancer survival outcomes** using a **multimodal dataset** integrating:
- **RNA expression** (`rs_`)
- **Copy number variations (CNVs)** (`cn_`)
- **Somatic mutations** (`mu_`)
- **Protein phosphorylation** (`pp_`)


The goal was twofold:
1. **Prediction** — Train an **XGBoost classifier** to distinguish between *living* and *deceased* patients.
2. **Interpretation** — Use **SHAP (SHapley Additive exPlanations)** to identify the most influential molecular features driving model predictions.

The work reflects my broader research interest in **precision oncology**, **functional genomics**, and **explainable AI** for **personalized healthcare**

## Limitation and Disclaimer
Certain clinically important breast cancer genes (e.g., **BRCA1**, **TP53**) were partially missing or incomplete in the dataset. For example, while `mu_TP53` and `rs_BRCA2` were available, `rs_BRCA1` was absent. This does not diminish the methodology’s validity as a **pipeline for molecular feature discovery and explainable modeling**.
---

##  Dataset Description

- **Source**: `brca_data_w_subtypes.csv`
- **Samples**: Breast cancer patients with annotated survival outcomes.
- **Target Variable**: `vital.status` (0 = Living, 1 = Deceased)
- **Feature Types**:
  - `rs_` → RNA-seq gene expression
  - `cn_` → Copy number variation
  - `mu_` → Mutation status
  - `pp_` → Protein phosphorylation
- **Metadata**: PR, ER, HER2 status; histological type

**Key Genes Used in Analysis**:
- **RNA expression**: `rs_TP53`, `rs_PIK3CA`, `rs_AKT1`, `rs_CLEC3A`, `rs_PLIN1`, `rs_SLC7A2`
- **CNVs**: `cn_BRCA2`, `cn_ERBB2`, `cn_AKT1`
- **Mutations**: `mu_TP53`, `mu_PIK3CA`, `mu_PTEN`
- **Phosphorylation**: `pp_Akt.pS473`, `pp_Akt.pT308`

## Methodology

### 1. **Data Preprocessing**
- Missing values imputed using **mean imputation**
- Standardized using **Z-score normalization**
- Feature subsets categorized by modality for downstream analysis

### 2. **Feature Mapping and Visualization**
- Plotted top 5 highly expressed genes
- Analyzed mutation frequency distribution
- Visualized relationships between gene expression and protein activity
- Explored correlations between key genomic features

### 3. **Dimensionality Reduction**
- Applied **PCA** to project key genes into 2D space
- Colored samples by survival status

### 4. **Model Training**
- Trained an **XGBoost classifier**
- Performed hyperparameter tuning via **GridSearchCV**
- Evaluated using classification metrics and a confusion matrix

### 5. **Model Interpretability**
- Computed **SHAP values** for top 100 test samples
- Generated:
  - SHAP **summary plot**
  - SHAP **dependence plot**
  - SHAP **force plot** (sample-level explanation)

---

##  Key Visualizations (Saved to `/figures/`)

| Visualization | Description |
|---------------|-------------|
| `target_distribution.png` | Survival status class balance |
| `top_expressed_genes.png` | Most highly expressed RNA-seq genes |
| `top_mutated_genes.png` | Mutation frequency by gene |
| `correlation_heatmap.png` | Correlation among key features |
| `pca_key_genes.png` | PCA plot of select RNA features |
| `confusion_matrix_heatmap.png` | Classification results |
| `shap_summary.png` | Global feature importance |
| `shap_dependence.png` | Dependence plot for top feature |
| `shap_force_sample0.png` | Sample-level interpretability |

---

## 🔍 Model Performance
- **Best Model**: Tuned XGBoost
- **Metrics**:
  - Accuracy: 0.89
  - Precision / Recall / F1 [0]: 0.90/0.99/0.94
  - Precision / Recall / F1 [1]: 0.83/0.26/0.40

---

## ⚙️ Installation and Usage

### Prerequisites
- Python 3.8+
- Dependencies listed in `requirements.txt`

### Setup Instructions

```bash
# Clone repository and navigate
git clone <your-repo-url>
cd brca-survival-prediction

# Install required libraries
pip install -r requirements.txt

# Run the pipeline
python main.py
```

All figures and the trained model (`survival_outcome_model.pkl`) will be saved to the project directory.

---

## Applications and Significance

- **Precision Medicine**: Identifies molecular patterns associated with survival outcomes
- **Explainable AI**: Enhances transparency of genomic ML models
- **Research Utility**: Framework adaptable to other cancers and omics data

---

## 📚 References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP for Explainable AI](https://shap.readthedocs.io/en/latest/)
- [Scikit-learn: Model Evaluation Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- Breast Cancer Genomics Literature (e.g., TP53, BRCA1, PIK3CA involvement in survival)

---
