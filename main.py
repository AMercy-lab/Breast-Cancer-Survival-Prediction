import pandas as pd
import numpy as np
import os
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

# Set visual style
sns.set(style="whitegrid")

# Create figures directory
os.makedirs("figures", exist_ok=True)

# Load dataset
DATA_PATH = "./archive/brca_data_w_subtypes.csv"
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset with shape: {df.shape}")

# Plot target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=df['vital.status'].astype(int))
plt.title("Target Class Distribution (0: Living, 1: Deceased)")
plt.xlabel("Vital Status")
plt.ylabel("Count")
plt.savefig("figures/target_distribution.png")
plt.close()
print("Target distribution plot saved.")

# Identify feature modalities
metadata_cols = ['vital.status', 'PR.Status',
                 'ER.Status', 'HER2.Final.Status', 'histological.type']
rna_features = [col for col in df.columns if col.startswith("rs_")]
cnv_features = [col for col in df.columns if col.startswith("cn_")]
mu_features = [col for col in df.columns if col.startswith("mu_")]
protein_features = [col for col in df.columns if col.startswith("pp_")]
all_features = rna_features + cnv_features + mu_features + protein_features

# Map feature names: strip prefixes for rs_, cn_, mu_, pp_
feature_names = []
for f in all_features:
    if f.startswith(('rs_', 'cn_', 'mu_', 'pp_')):
        gene_name = f.replace('rs_', '').replace(
            'cn_', '').replace('mu_', '').replace('pp_', '')
        feature_names.append(gene_name)
    else:
        feature_names.append(f)

# Preprocessing
X = df[all_features].copy()
X = X.apply(pd.to_numeric, errors='coerce')
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
y = df['vital.status'].astype(int).values
print(f"Combined feature shape: {X_scaled.shape}")
print(f"Label distribution: {np.bincount(y)}")

# Feature Mapping
#available key_genes on the dataset rs_CLEC3A, rs_PLIN1, rs_SLC7A2
key_genes = ['rs_TP53', 'rs_BRCA1', 'rs_PIK3CA', 'rs_AKT1', 'rs_AKT2', 'rs_CLEC3A', 'rs_PLIN1', 'rs_SLC7A2']
key_cn = ['cn_BRCA2', 'cn_ERBB2', 'cn_AKT1', 'cn_AKT2', 'cn_TPSAB1']
key_mu = ['mu_TP53', 'mu_PIK3CA', 'mu_PTEN', 'mu_GATA3']
key_pp = ['pp_Akt.pS473','pp_Akt.pT308'] if 'pp_Akt.pS473' in protein_features else protein_features[:2]

# Gene Expression (rs_)
rs_stats = pd.DataFrame({
    'Mean': X[rna_features].mean(),
    'Variance': X[rna_features].var()
}).sort_values(by='Mean', ascending=False)
top_rs = rs_stats.head(5)
plt.figure(figsize=(8, 5))
sns.barplot(x=top_rs.index.map(
    lambda x: x.replace('rs_', '')), y=top_rs['Mean'])
plt.title("Top 5 Highly Expressed Genes")
plt.xlabel("Gene")
plt.ylabel("Mean Expression")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/top_expressed_genes.png")
plt.close()

# Copy Number Variation (cn_)
if 'cn_TPSAB1' in cnv_features:
    plt.figure(figsize=(8, 5))
    sns.histplot(X['cn_TPSAB1'].dropna(), bins=20, color='green')
    plt.title("Copy Number Variation Distribution for BRCA1")
    plt.xlabel("Copy Number Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("figures/cn_brca1_histogram.png")
    plt.close()

# Mutation Frequencies (mu_)
mu_freq = X[mu_features].mean() * 100
top_mu = mu_freq.sort_values(ascending=False).head(5)
plt.figure(figsize=(8, 5))
sns.barplot(x=top_mu.index.map(lambda x: x.replace('mu_', '')), y=top_mu)
plt.title("Top 5 Mutated Genes")
plt.xlabel("Gene")
plt.ylabel("Mutation Frequency (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/top_mutated_genes.png")
plt.close()

# Protein Phosphorylation (pp_)
pp_stats = pd.DataFrame({
    'Mean': X[protein_features].mean()
})
if 'pp_Akt.pS473' in protein_features and 'rs_CLEC3A' in rna_features:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=X['rs_CLEC3A'], y=X['pp_Akt.pS473'], hue=y, palette='Set1')
    plt.title("rs_CLEC3A Expression vs. pp_Akt.pS473 Phosphorylation")
    plt.xlabel("rs_CLEC3A Expression")
    plt.ylabel("pp_Akt.pS473 Level")
    plt.legend(title="Vital Status", labels=["Living", "Deceased"])
    plt.tight_layout()
    plt.savefig("figures/akt_scatter.png")
    plt.close()

# Correlation Heatmap
key_features = [f for f in key_genes +
                key_cn + key_mu + key_pp if f in X.columns]
corr_matrix = X[key_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title("Correlation Matrix for Key Features")
plt.tight_layout()
plt.savefig("figures/correlation_heatmap.png")
plt.close()

# PCA for Key Genes
if all(f in X.columns for f in key_genes):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X[key_genes])
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='Set1')
    plt.title("PCA Projection of Key Genes (TP53, BRCA1, PIK3CA, PTEN)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Vital Status", labels=["Living", "Deceased"])
    plt.tight_layout()
    plt.savefig("figures/pca_key_genes.png")
    plt.close()

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Model Training
model = XGBClassifier(eval_metric="logloss", random_state=42)
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1, 0.3]
}
grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", verbose=1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print(f"Best parameters: {grid.best_params_}")

# Evaluation
y_pred = best_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[
            "Living", "Deceased"], yticklabels=["Living", "Deceased"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("figures/confusion_matrix_heatmap.png")
plt.close()
print("Confusion matrix heatmap saved.")

# Save Model
joblib.dump(best_model, "survival_outcome_model.pkl")
print("Model saved to survival_outcome_model.pkl")

# SHAP Explainability
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test[:100])
shap.summary_plot(shap_values, X_test[:100],
    feature_names=feature_names, show=False)
# SHAP Summary Plotshap_values, X_test[:100],
plt.title("SHAP Feature Importance (Top 100 Test Samples)")
plt.tight_layout()
plt.savefig("figures/shap_summary.png", bbox_inches="tight")
plt.close()

# SHAP Dependence Plot
top_feat_idx = np.abs(shap_values).mean(0).argmax()
shap.dependence_plot(top_feat_idx, shap_values,
                     X_test[:100], feature_names=feature_names, show=False)
plt.title(f"SHAP Dependence Plot ({feature_names[top_feat_idx]})")
plt.tight_layout()
plt.savefig("figures/shap_dependence.png", bbox_inches="tight")
plt.close()

# SHAP Force Plot
shap.force_plot(explainer.expected_value,
                shap_values[0], X_test[:1], feature_names=feature_names, matplotlib=True, show=False)
plt.tight_layout()
plt.savefig("figures/shap_force_sample0.png", bbox_inches="tight")
plt.close()

# Feature Importance Plot
importance = best_model.feature_importances_
top_indices = np.argsort(importance)[-10:]
top_features = [feature_names[i] for i in top_indices]
top_values = importance[top_indices]
plt.figure(figsize=(10, 5))
sns.barplot(x=top_features, y=top_values)
plt.title("Top 10 Feature Importances from XGBoost")
plt.ylabel("Importance Score")
plt.xlabel("Feature (Gene/Protein)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/top10_features.png")
plt.close()
print("Top 10 feature importance plot saved.")

# Summary Report
print("\n=== BRCA Feature Mapping Report ===")
print("\nTop 5 Gene Expression Statistics:")
print(rs_stats.head())
print("\nMutation Frequencies (%):")
print(mu_freq[key_mu])
print("\nProtein Phosphorylation Means:")
print(pp_stats.loc[key_pp])
print("\nKey Feature Correlations:")
print(corr_matrix)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
