# Breast Cancer Survival Prediction and Feature Analysis

## Overview
This project analyzes the `brca_data_w_subtypes.csv` dataset, which contains multi-omics data (RNA expression, copy number variations, mutations, and protein phosphorylation) for breast cancer samples. The script performs exploratory data analysis (EDA), trains an XGBoost model to predict survival outcome (`vital.status`), and uses SHAP (SHapley Additive exPlanations) to interpret feature importance. Visualizations include target distribution, feature statistics, PCA, correlation heatmaps, and SHAP plots.

## Requirements
- Python 3.8+
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - shap
  - matplotlib
  - seaborn
  - joblib
- Install dependencies:
  ```bash
  pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn joblib
  ```

## Dataset
- **File**: `brca_data_w_subtypes.csv`
- **Description**: Contains 674 features (RNA expression (`rs_`), copy number variations (`cn_`), mutations (`mu_`), protein phosphorylation (`pp_`)) and clinical metadata (`vital.status`, `PR.Status`, `ER.Status`, `HER2.Final.Status`, `histological.type`).
- **Target**: `vital.status` (0: Living, 1: Deceased).
- **Note**: Some expected features (e.g., `rs_BRCA1`, `cn_BRCA1`, `mu_BRCA1`, `mu_PTEN`) are missing from the dataset. Alternatives like `rs_BRCA2`, `mu_TP53`, and `cn_BRCA2` are used.

## Features
- **RNA Expression (`rs_`)**: Gene expression levels (e.g., `rs_AKT1`, `rs_TP53`).
- **Copy Number Variations (`cn_`)**: Genomic alterations (e.g., `cn_BRCA2`, `cn_ERBB2`).
- **Mutations (`mu_`)**: Mutation status (e.g., `mu_TP53`, `mu_PIK3CA`).
- **Protein Phosphorylation (`pp_`)**: Phosphorylation levels (e.g., `pp_Akt.pS473`).
- **Key Features Used**:
  - RNA: `rs_TP53`, `rs_PIK3CA`, `rs_AKT1`, `rs_AKT2`, `rs_CLEC3A`, `rs_PLIN1`, `rs_SLC7A2`
  - CNV: `cn_BRCA2`, `cn_ERBB2`, `cn_AKT1`, `cn_AKT2`, `cn_TPSAB1`
  - Mutation: `mu_TP53`, `mu_PIK3CA`, `mu_GATA3`
  - Phosphorylation: `pp_Akt.pS473`, `pp_Akt.pT308` (if available, else first two `pp_` features)

## Script Functionality
1. **Data Loading and EDA**:
   - Loads the dataset and plots target distribution (`vital.status`).
   - Computes statistics for RNA expression, CNVs, mutations, and phosphorylation.
   - Visualizes top 5 expressed genes, mutation frequencies, and phosphorylation correlations.
   - Generates a correlation heatmap for key features.
   - Performs PCA on key RNA genes (if all are present).

2. **Preprocessing**:
   - Converts features to numeric, imputes missing values (mean strategy), and standardizes data.
   - Splits data into training (80%) and test (20%) sets with stratification.

3. **Model Training**:
   - Uses XGBoost to predict `vital.status`.
   - Performs GridSearchCV with 3-fold cross-validation to tune hyperparameters (`n_estimators`, `max_depth`, `learning_rate`).
   - Evaluates model with classification report and confusion matrix.

4. **SHAP Analysis**:
   - Computes SHAP values for the top 100 test samples.
   - Generates SHAP summary, dependence, and force plots to interpret feature importance.

5. **Outputs**:
   - Saves visualizations to the `figures/` directory.
   - Saves the trained model to `survival_outcome_model.pkl`.
   - Prints a summary report with feature statistics, correlations, and model accuracy.

## File Structure
```
project_directory/
│
├── archive/
│   └── brca_data_w_subtypes.csv  # Input dataset
├── figures/                      # Output visualizations
│   ├── target_distribution.png
│   ├── top_expressed_genes.png
│   ├── cn_brca1_histogram.png    # If cn_TPSAB1 is present
│   ├── top_mutated_genes.png
│   ├── akt_scatter.png           # If pp_Akt.pS473 and rs_CLEC3A are present
│   ├── correlation_heatmap.png
│   ├── pca_key_genes.png         # If all key genes are present
│   ├── confusion_matrix_heatmap.png
│   ├── shap_summary.png
│   ├── shap_dependence.png
│   ├── shap_force_sample0.png
│   ├── top10_features.png
├── survival_outcome_model.pkl    # Trained XGBoost model
├── brca_analysis.py             # Main script
└── README.md                    # This file
```

## Usage
1. Place `brca_data_w_subtypes.csv` in the `archive/` directory.
2. Ensure all required libraries are installed.
3. Run the script:
   ```bash
   python brca_analysis.py
   ```
4. Check the `figures/` directory for visualizations and `survival_outcome_model.pkl` for the trained model.

## Notes
- **Missing Features**: The script checks for key features (e.g., `rs_BRCA1`, `cn_BRCA1`, `mu_PTEN`). If missing, it uses available features (e.g., `rs_BRCA2`, `mu_TP53`). Verify feature availability with:
  ```python
  print([f for f in key_genes if f not in df.columns])  # Lists missing key genes
  ```
- **Error Handling**: The PCA and CNV histogram sections are conditional to avoid errors if features are missing.
- **SHAP Issue**: Ensure `shap.summary_plot(shap_values, X_test[:100], ...)` is used (not `shap_values(X_test[:100])`), as `shap_values` is a NumPy array, not a function.
- **Customization**: Adjust `key_genes`, `key_cn`, `key_mu`, and `key_pp` lists based on available features or analysis focus.

## Outputs
- **Visualizations**: Saved in `figures/` (e.g., SHAP plots, PCA, confusion matrix).
- **Model**: Saved as `survival_outcome_model.pkl`.
- **Console Output**: Includes feature statistics, model performance, and correlations.

## Known Issues
- Some key features (e.g., `rs_BRCA1`, `cn_BRCA1`, `mu_PTEN`) may be missing, affecting PCA and correlation analyses. Alternatives (e.g., `rs_BRCA2`, `cn_BRCA2`) are used.
- The script assumes `vital.status` is binary (0/1). If non-numeric, ensure proper encoding.
- SHAP force plots may require adjustment for multi-class problems (not applicable here, as `vital.status` is binary).

## Future Improvements
- Add feature selection to reduce noise (e.g., recursive feature elimination).
- Handle class imbalance with SMOTE if `vital.status` is imbalanced.
- Expand PCA to include more feature types (e.g., `key_mu`, `key_pp`).
- Test alternative models (e.g., Random Forest, LightGBM).

## Contact
For issues or questions, please contact [your contact info or leave blank].