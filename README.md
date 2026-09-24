# Customer Churn Analysis and Prediction Using Data Analytics and Machine Learning

## Project Overview
This project analyzes customer data to identify patterns associated with churn and builds a machine learning model to predict whether a customer is likely to leave. It is built as part of the AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship 2026.

## Problem Statement
Customer churn is a critical metric for businesses. Identifying factors that lead to customer attrition allows companies to proactively address issues and improve retention rates.

## Objectives
- Perform Exploratory Data Analysis (EDA) on customer data.
- Identify key features influencing churn.
- Build and evaluate a Machine Learning model (Logistic Regression) to predict churn.

## Dataset
- **Dataset name:** Telco Customer Churn
- **Dataset source:** Kaggle / IBM Sample Datasets
- **Dataset link:** https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- **Description:** Contains demographics, services, account information, and churn status.
- **Records:** Over 7000 customer records.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Structure
```
customer-churn-ai-project/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   └── Customer_Churn_Analysis_and_Prediction.ipynb
│
├── outputs/
│   └── (Generated charts and metrics)
│
├── Customer_Churn_Analysis_and_Prediction.py
├── build_artifacts.py
├── requirements.txt
├── README.md
└── Customer_Churn_ProjectReport.docx
```

## Installation

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Project
1. Ensure the dataset `WA_Fn-UseC_-Telco-Customer-Churn.csv` is in the `data/` folder.
2. Run the main analysis script to generate charts and metrics:
   ```bash
   python Customer_Churn_Analysis_and_Prediction.py
   ```
3. (Optional) Run the artifact builder to regenerate the notebook and report:
   ```bash
   python build_artifacts.py
   ```
4. Open the Jupyter Notebook to explore the code interactively:
   ```bash
   jupyter notebook notebooks/Customer_Churn_Analysis_and_Prediction.ipynb
   ```

## Machine Learning
The project uses **Logistic Regression** as the primary predictive model. A `ColumnTransformer` is used to scale numerical features and encode categorical features.

## Evaluation Metrics
- **Accuracy**: The proportion of correct predictions.
- **Precision**: The proportion of positive identifications that were actually correct.
- **Recall**: The proportion of actual positives that were identified correctly (crucial for churn).
- **F1-score**: The harmonic mean of precision and recall.
- **ROC-AUC**: The ability of the model to distinguish between classes.

## Key Findings
- Month-to-month contracts are highly associated with customer churn.
- Customers with longer tenure are less likely to churn.
- Higher monthly charges are generally associated with a higher risk of churn.
- Electronic checks as a payment method show a higher association with churn.

## Future Enhancements
- Hyperparameter tuning using GridSearchCV.
- Training advanced tree-based ensemble models (e.g., Random Forest, XGBoost).
- Building a real-time web dashboard using Flask, FastAPI, or Streamlit.

## Author
Name: J S Caitlyn Mary
Internship: IBM SkillsBuild Data Analytics with AI
Program: AICTE / BharatCares
