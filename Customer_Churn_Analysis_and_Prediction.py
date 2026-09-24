import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import os
import json

def run_analysis():
    print("Starting Customer Churn Analysis and Prediction...")
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # 1. Load Dataset
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print(f"Dataset loaded. Shape: {df.shape}")
    
    # 2. Data Cleaning
    # Replace empty spaces with NaN in TotalCharges and drop them or fill them
    df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    # Since only 11 rows have missing TotalCharges, we can drop them
    df = df.dropna(subset=['TotalCharges'])
    print(f"Dataset shape after cleaning: {df.shape}")
    
    # Drop customerID as it's not a useful feature for ML
    df = df.drop('customerID', axis=1)
    
    # 3. Exploratory Data Analysis & Visualizations
    sns.set_theme(style="whitegrid")
    
    # Churn distribution
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='Churn', palette='Set2')
    plt.title('Churn Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.savefig('outputs/churn_distribution.png')
    plt.close()
    
    # Churn by Contract Type
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x='Contract', hue='Churn', palette='Set2')
    plt.title('Churn by Contract Type')
    plt.xlabel('Contract Type')
    plt.ylabel('Count')
    plt.savefig('outputs/churn_by_contract.png')
    plt.close()
    
    # Monthly Charges Distribution
    plt.figure(figsize=(8,5))
    sns.histplot(data=df, x='MonthlyCharges', hue='Churn', multiple="stack", palette='Set2', bins=30)
    plt.title('Monthly Charges Distribution by Churn')
    plt.xlabel('Monthly Charges')
    plt.ylabel('Count')
    plt.savefig('outputs/monthly_charges_dist.png')
    plt.close()
    
    # Churn by Payment Method
    plt.figure(figsize=(10,5))
    sns.countplot(data=df, x='PaymentMethod', hue='Churn', palette='Set2')
    plt.title('Churn by Payment Method')
    plt.xlabel('Payment Method')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/churn_by_payment.png')
    plt.close()
    
    # Correlation heatmap for numerical variables
    plt.figure(figsize=(8,6))
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap (Numerical Features)')
    plt.savefig('outputs/correlation_heatmap.png')
    plt.close()
    
    # 4. Data Preprocessing
    # Target variable encoding
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Identify categorical and numerical columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
        ]
    )
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # 5. Model Training (Logistic Regression)
    lr_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    lr_model.fit(X_train, y_train)
    
    # 6. Model Evaluation
    y_pred = lr_model.predict(X_test)
    y_pred_prob = lr_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1_Score': f1_score(y_test, y_pred),
        'ROC_AUC': roc_auc_score(y_test, y_pred_prob)
    }
    
    print("\n--- Logistic Regression Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix - Logistic Regression')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {metrics["ROC_AUC"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig('outputs/roc_curve.png')
    plt.close()
    
    # 7. Feature Importance (Logistic Regression Coefficients)
    feature_names = num_cols + list(lr_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols))
    coefficients = lr_model.named_steps['classifier'].coef_[0]
    
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    feat_imp['Abs_Coefficient'] = feat_imp['Coefficient'].abs()
    feat_imp = feat_imp.sort_values(by='Abs_Coefficient', ascending=False).head(10)
    
    plt.figure(figsize=(10,6))
    sns.barplot(data=feat_imp, x='Coefficient', y='Feature', palette='coolwarm')
    plt.title('Top 10 Feature Coefficients (Logistic Regression)')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png')
    plt.close()
    
    # 8. Save Metrics to JSON for report generation
    with open('outputs/metrics.json', 'w') as f:
        json.dump(metrics, f)
        
    print("\nAnalysis complete. Visualizations saved in 'outputs' folder.")
    
    # 9. Example Prediction
    example_data = pd.DataFrame([X_test.iloc[0]])
    example_pred = lr_model.predict(example_data)
    example_prob = lr_model.predict_proba(example_data)[:, 1]
    
    print(f"\n--- Example Prediction ---")
    print(f"Input features:\n{example_data.to_dict(orient='records')[0]}")
    print(f"Predicted Churn: {'Yes' if example_pred[0] == 1 else 'No'}")
    print(f"Churn Probability: {example_prob[0]:.4f}")

if __name__ == "__main__":
    run_analysis()
