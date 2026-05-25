# Loan Approval Prediction System

## Project Overview

This project is an end-to-end Machine Learning application that predicts whether a loan will be approved or rejected based on applicant details.

The project includes:
- Data preprocessing
- Feature engineering
- Model training
- Hyperparameter tuning
- Model evaluation
- Streamlit web application

---

# Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Streamlit

---

# Machine Learning Workflow

## 1. Data Preprocessing

The following preprocessing steps were performed:

- Missing value handling
- Outlier handling using IQR method
- Skewness handling using log transformation
- One Hot Encoding
- Feature Scaling using StandardScaler

---

## 2. Feature Engineering

Created a new feature:

```python
Has_Coapplicant
```

This feature identifies whether a co-applicant exists or not.

---

## 3. Models Used

The following machine learning models were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

---

## 4. Model Evaluation

Techniques used:

- Cross Validation
- GridSearchCV
- Accuracy Score
- Classification Report

---

# Project Structure

```text
Loan-Approval-Project/

│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── Loan.csv
│
├── models/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   └── Loanproject.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── train.py
│   ├── train_pipeline.py
│   └── utils.py
│
├── requirements.txt
├── README.md
```

---

# How to Run the Project

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2: Train the Model

```bash
python -m src.train_pipeline
```

This creates:
- model.pkl
- preprocessor.pkl

inside the models folder.

---

## Step 3: Run Streamlit App

```bash
streamlit run app/app.py
```

---

# Model Performance

The final model achieved approximately:

```text
85% Accuracy
```

using GridSearchCV tuned SVM model.

---

# Features Used

- Gender
- Married
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History

---

# Future Improvements

- Deployment on cloud platform
- Advanced feature engineering
- Pipeline automation
- Better UI design
- Additional ML models

---

# Author

Mancha.Venkata Navya Tej

B.Tech CSE (AI/ML) Student  
Machine Learning Project