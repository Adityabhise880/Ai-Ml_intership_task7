# ===================================================
# TASK 7: TITANIC SURVIVAL PREDICTION
# Logistic Regression - Complete Solution
# ===
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TITANIC SURVIVAL PREDICTION - LOGISTIC REGRESSION")
print("="*60)

# ===================================================
# STEP 1: LOAD DATASET
# ===================================================
print("\n[1/9] Loading Titanic dataset...")
df = sns.load_dataset('titanic')
print(f"[OK] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ===================================================
# STEP 2: HANDLE MISSING VALUES
# ===================================================
print("\n[2/9] Handling missing values...")
print(f"Missing values before:\n{df.isnull().sum()}\n")

# Fill Age with median
df['age'].fillna(df['age'].median(), inplace=True)

# Fill Embarked with mode
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Drop deck column (too many missing)
df = df.drop('deck', axis=1)

print(f"[OK] Missing values handled")
print(f"Missing values after:\n{df.isnull().sum()}\n")

# ===================================================
# STEP 3: REMOVE UNNECESSARY COLUMNS
# ===================================================
print("\n[3/9] Removing unnecessary columns...")

# Keep only important columns
columns_to_keep = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
df = df[columns_to_keep]

print(f"[OK] Columns kept: {df.columns.tolist()}")

# ===================================================
# STEP 4: ENCODE CATEGORICAL FEATURES
# ===================================================
print("\n[4/9] Encoding categorical features...")

# Encode sex: male=1, female=0
df['sex'] = df['sex'].map({'male': 1, 'female': 0})

# One-Hot Encoding for embarked (C, Q, S)
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

print(f"[OK] Encoding complete")
print(f"Final columns: {df.columns.tolist()}")

# ===================================================
# STEP 5: SPLIT FEATURES AND TARGET
# ===================================================
print("\n[5/9] Separating features and target...")

X = df.drop('survived', axis=1)  # Features
y = df['survived']  # Target

print(f"[OK] Features shape: {X.shape}")
print(f"[OK] Target shape: {y.shape}")

# ===================================================
# STEP 6: STANDARDIZE FEATURES
# ===================================================
print("\n[6/9] Applying StandardScaler...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"[OK] Scaling complete")

# ===================================================
# STEP 7: SPLIT INTO TRAIN-TEST
# ===================================================
print("\n[7/9] Splitting data into train and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[OK] Training set: {X_train.shape[0]} samples")
print(f"[OK] Test set: {X_test.shape[0]} samples")

# ===================================================
# STEP 8: TRAIN LOGISTIC REGRESSION MODEL
# ===================================================
print("\n[8/9] Training Logistic Regression model...")

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print(f"[OK] Model trained successfully!")

# ===================================================
# STEP 9: MAKE PREDICTIONS AND EVALUATE
# ===================================================
print("\n[9/9] Making predictions and evaluating...")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("="*60)

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived']))

# ===================================================
# GENERATE CONFUSION MATRIX IMAGE
# ===================================================
print("\n[CHART] Generating Confusion Matrix...")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Did Not Survive', 'Survived'],
            yticklabels=['Did Not Survive', 'Survived'],
            cbar=True, linewidths=1, linecolor='black')
plt.title('Confusion Matrix - Titanic Survival Prediction', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight', facecolor='white')
print("[OK] Confusion matrix saved as 'confusion_matrix.png'")
plt.close()

# ===================================================
# GENERATE ROC CURVE AND AUC SCORE
# ===================================================
print("\n[GRAPH] Generating ROC Curve...")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curve - Titanic Survival Prediction', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] ROC curve saved as 'roc_curve.png'")
print(f"[OK] AUC Score: {roc_auc:.4f}")
plt.close()

# ===================================================
# FINAL SUMMARY
# ===================================================
print("\n" + "="*60)
print("[DONE] TASK COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nDeliverables Generated:")
print("1. [OK] confusion_matrix.png")
print("2. [OK] roc_curve.png")
print(f"3. [OK] Model Accuracy: {accuracy:.4f}")
print(f"4. [OK] AUC Score: {roc_auc:.4f}")
print("\nNext Steps:")
print("-> Create GitHub repository")
print("-> Upload code and images")
print("-> Create README.md")
print("-> Submit your repository link")
print("="*60)