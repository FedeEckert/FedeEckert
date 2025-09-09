import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\paige\\Dropbox\\FinTech Group 2025\\Data Analytics 3.25\\New Code as of Mar 21\\financial_crypto_datasetLOG.csv")

data_for_pca = df.drop(columns=['Date'])

# Scale the data
scaler = StandardScaler(with_mean=True, with_std=True)
data_scaled = scaler.fit_transform(data_for_pca)

# Confirm scaling
print("Scaled Data Mean (should be ~0):", data_scaled.mean(axis=0))
print("Scaled Data Std Dev (should be ~1):", data_scaled.std(axis=0))

# Perform PCA
pca = PCA()
pca.fit(data_scaled)
scores = pca.transform(data_scaled)

# Scree Plot with 70% line and PC6 highlight
pve = pca.explained_variance_ratio_
cumulative_pve = np.cumsum(pve)

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(pve) + 1), pve, alpha=0.6, color='skyblue', edgecolor='black', label="PVE")
plt.plot(range(1, len(pve) + 1), cumulative_pve, marker="o", color='orange', linewidth=2.5, label="Cumulative PVE")
plt.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label="70% Variance Threshold")
plt.axvline(x=6, color='green', linestyle='--', linewidth=2, label="Chosen limit (PC6)")
plt.xlabel("Principal Components", fontsize=12)
plt.ylabel("Variance Explained", fontsize=12)
plt.title("Scree Plot with 70% Line and PC6 Selection", fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.show()

# Print variance explained by first 3 PCs
var_pc1_3 = cumulative_pve[2]
print(f"\nThe first 3 principal components explain {var_pc1_3:.2%} of the total variance.\n")

# Display PCA Loadings for PC1, PC2, PC3
loadings_df = pd.DataFrame(pca.components_.T, 
                           columns=[f'PC{i+1}' for i in range(len(pca.components_))],
                           index=data_for_pca.columns)
print("PCA Loadings (first 3 PCs):")
print(loadings_df.iloc[:, :3].round(4))

# Biplot for PC1 vs PC2
i, j = 0, 1
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(scores[:, i], scores[:, j], alpha=0.6, label='Observations')
for k in range(pca.components_.shape[1]):
    ax.arrow(0, 0, pca.components_[i, k] * 2, pca.components_[j, k] * 2, color='r', alpha=0.75, head_width=0.05)
    ax.text(pca.components_[i, k] * 2.4, pca.components_[j, k] * 2.4, data_for_pca.columns[k], color='g', fontsize=10, ha='center', va='center')
ax.set_xlabel(f'PC{i+1}')
ax.set_ylabel(f'PC{j+1}')
ax.set_title('Biplot: PC1 vs PC2')
ax.grid(True)
ax.legend()
plt.show()

# Biplot for PC1 vs PC3
i, j = 0, 2
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(scores[:, i], scores[:, j], alpha=0.6, label='Observations')
for k in range(pca.components_.shape[1]):
    ax.arrow(0, 0, pca.components_[i, k] * 2, pca.components_[j, k] * 2, color='r', alpha=0.75, head_width=0.05)
    ax.text(pca.components_[i, k] * 2.4, pca.components_[j, k] * 2.4, data_for_pca.columns[k], color='g', fontsize=10, ha='center', va='center')
ax.set_xlabel(f'PC{i+1}')
ax.set_ylabel(f'PC{j+1}')
ax.set_title('Biplot: PC1 vs PC3')
ax.grid(True)
ax.legend()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 10))
corr_matrix = data_for_pca.corr()

# correlation heatmap
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=True, fmt=".2f", linewidths=0.5, cbar_kws={'label': 'Correlation'})
plt.title("Correlation Matrix Heatmap of Financial & Crypto Features", fontsize=16, fontweight='bold')
plt.xticks(rotation=45, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()

from sklearn.preprocessing import StandardScaler

# Extract PC1 scores and BTC returns
fci = pd.Series(scores[:, 0], index=pd.to_datetime(df['Date']), name="FCI_PC1")
btc_series = pd.Series(df['BTC_Volatility'].values, index=pd.to_datetime(df['Date']), name="BTC_Volatility")

# Standardize both series
scaler = StandardScaler()
fci_scaled = pd.Series(scaler.fit_transform(fci.values.reshape(-1, 1)).flatten(), index=fci.index, name="FCI_Scaled")
btc_scaled = pd.Series(scaler.fit_transform(btc_series.values.reshape(-1, 1)).flatten(), index=btc_series.index, name="BTC_Volatility_Scaled")

# Plot both on the same scale
plt.figure(figsize=(14, 6))
plt.plot(fci_scaled, label="Financial Conditions Index (PC1)", color='blue', linewidth=1.5)
plt.plot(btc_scaled, label="BTC Volatility (scaled)", color='orange', alpha=0.9, linewidth=1.5)

plt.title("Standardized Financial Conditions Index (PC1) vs. BTC Volatility", fontsize=16, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Standardized Values")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

df = pd.read_csv("financial_crypto_datasetLOG.csv")
df['Date'] = pd.to_datetime(df['Date'])

data_for_pca = df.drop(columns=['Date'])
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_for_pca)

pca = PCA()
scores = pca.fit_transform(data_scaled)

# Create classification target based on BTC volatility median
median_vol = df['BTC_Volatility'].median()
df['Vol_Class'] = (df['BTC_Volatility'] > median_vol).astype(int)

#Prepare data and split
X = pd.DataFrame(scores[:, :6], columns=[f'PC{i+1}' for i in range(6)])
y = df['Vol_Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_test_prob_log = log_reg.predict_proba(X_test)[:, 1]
y_test_class_log = log_reg.predict(X_test)
auc_score_log = roc_auc_score(y_test, y_test_prob_log)
print(f"Logistic Regression (Test Set) - AUC: {auc_score_log:.4f}")
print(confusion_matrix(y_test, y_test_class_log))

fpr_log, tpr_log, _ = roc_curve(y_test, y_test_prob_log)
plt.figure(figsize=(7,5))
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {auc_score_log:.4f})', color='blue')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression (Test Set)')
plt.legend()
plt.grid()
plt.show()

# LDA
lda = LDA()
lda.fit(X_train, y_train)
y_test_prob_lda = lda.predict_proba(X_test)[:, 1]
y_test_class_lda = (y_test_prob_lda >= 0.5).astype(int)
auc_score_lda = roc_auc_score(y_test, y_test_prob_lda)
print(f"LDA (Test Set, cutoff=0.5) - AUC: {auc_score_lda:.4f}")
print(confusion_matrix(y_test, y_test_class_lda))

fpr_lda, tpr_lda, _ = roc_curve(y_test, y_test_prob_lda)
plt.figure(figsize=(7,5))
plt.plot(fpr_lda, tpr_lda, label=f'LDA (AUC = {auc_score_lda:.4f})', color='green')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - LDA (Test Set)')
plt.legend()
plt.grid()
plt.show()

# QDA
qda = QDA()
qda.fit(X_train, y_train)
y_test_prob_qda = qda.predict_proba(X_test)[:, 1]
y_test_class_qda = qda.predict(X_test)
auc_score_qda = roc_auc_score(y_test, y_test_prob_qda)
print(f"QDA (Test Set) - AUC: {auc_score_qda:.4f}")
print(confusion_matrix(y_test, y_test_class_qda))

fpr_qda, tpr_qda, _ = roc_curve(y_test, y_test_prob_qda)
plt.figure(figsize=(7,5))
plt.plot(fpr_qda, tpr_qda, label=f'QDA (AUC = {auc_score_qda:.4f})', color='purple')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - QDA (Test Set)')
plt.legend()
plt.grid()
plt.show()


# Combined ROC comparison plot
plt.figure(figsize=(10, 7))

fpr_log, tpr_log, _ = roc_curve(y_test, y_test_prob_log)
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {auc_score_log:.4f})', color='blue')

fpr_lda, tpr_lda, _ = roc_curve(y_test, y_test_prob_lda)
plt.plot(fpr_lda, tpr_lda, label=f'LDA (AUC = {auc_score_lda:.4f})', color='green')

fpr_qda, tpr_qda, _ = roc_curve(y_test, y_test_prob_qda)
plt.plot(fpr_qda, tpr_qda, label=f'QDA (AUC = {auc_score_qda:.4f})', color='purple')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison on Test Set')
plt.legend()
plt.grid()
plt.show()

# Function to plot confusion matrix heatmap
def plot_conf_matrix_heatmap(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

# Confusion matrix plots on test set
plot_conf_matrix_heatmap(y_test, y_test_class_log, 'Logistic Regression Confusion Matrix (Test Set)')
plot_conf_matrix_heatmap(y_test, y_test_class_lda, 'LDA Confusion Matrix (Test Set)')
plot_conf_matrix_heatmap(y_test, y_test_class_qda, 'QDA Confusion Matrix (Test Set)')

