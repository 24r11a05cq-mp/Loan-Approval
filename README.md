# Loan Approval Prediction Project

This project implements a complete end-to-end Machine Learning pipeline in Python to predict loan approvals based on applicant profiles. It is designed to demonstrate data loading, Exploratory Data Analysis (EDA), missing value imputation, categorical encoding, feature scaling, model training, evaluation comparison, and testing on custom sample profiles.

---

## Directory Structure

```text
loan_approval/
│
├── generate_data.py       # Generates the synthetic loan_data.csv dataset
├── loan_prediction.py     # Main ML pipeline (EDA, preprocessing, training, evaluation)
├── requirements.txt       # Dependencies (pandas, scikit-learn, numpy, matplotlib)
├── README.md              # Project documentation (this file)
│
# Generated files after running the scripts:
├── loan_data.csv          # The generated synthetic dataset
└── model_comparison.png   # Combined chart showing model comparison & confusion matrices
```

---

## Installation & Setup

1. **Prerequisites**: Ensure you have Python installed on your system.
2. **Install Dependencies**: Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### Step 1: Generate the Dataset
Create the synthetic dataset by running the generator script. This creates a realistic `loan_data.csv` file with 1,000 samples and some introduced missing values.
```bash
python generate_data.py
```

### Step 2: Run the Machine Learning Pipeline
Train, evaluate, and test the models by running the main prediction pipeline. This will print the summary metrics to the console and save the comparison plots.
```bash
python loan_prediction.py
```

---

## Dataset Schema

The generated dataset `loan_data.csv` contains 1,000 applicant records with the following features:

| Column Name | Data Type | Description | Missing Values |
|---|---|---|---|
| `ApplicantID` | String | Unique Identifier for each applicant | None |
| `Age` | Integer | Age of applicant (21 - 70) | None |
| `Gender` | Categorical | Gender (Male / Female) | None |
| `Married` | Categorical | Marital status (Yes / No) | 3.0% (introduced NaNs) |
| `Education` | Categorical | Education level (Graduate / Not Graduate) | None |
| `SelfEmployed` | Categorical | Employment status (Yes / No) | 4.0% (introduced NaNs) |
| `AnnualIncome` | Float | Annual income in USD ($20,000 - $300,000) | None |
| `CreditScore` | Float | Credit Score (300 - 850) | 5.0% (introduced NaNs) |
| `LoanAmount` | Float | Loan requested in USD ($5,000 - $600,000) | None |
| `YearsAtJob` | Float | Years at current job (0.0 - 40.0) | 5.0% (introduced NaNs) |
| `LoanApproved` | Binary | Target Variable: `1` (Approved) or `0` (Rejected) | None |

---

## Machine Learning Pipeline Steps

1. **Data Loading**: Loads `loan_data.csv` using `pandas`.
2. **EDA**: Displays data dimensions, head rows, and statistical descriptions.
3. **Imputation**: 
   - Numerical columns (`CreditScore`, `YearsAtJob`) are imputed using their **median** value (robust to outliers).
   - Categorical columns (`Married`, `SelfEmployed`) are imputed using their **mode** (most frequent class).
4. **Encoding**: Converts categorical columns (`Gender`, `Married`, `Education`, `SelfEmployed`) to numeric variables using `LabelEncoder`.
5. **Feature Scaling**: Scales numerical variables (`Age`, `AnnualIncome`, `CreditScore`, `LoanAmount`, `YearsAtJob`) using `StandardScaler` to bring them onto the same scale.
6. **Data Split**: Splits the dataset into **80% Training** and **20% Testing** sets (using stratified splitting to maintain class balance).
7. **Model Training**: Trains 3 standard scikit-learn models:
   - **Logistic Regression** (Linear Classifier)
   - **Decision Tree Classifier** (Tree-based model, capped at max_depth=5 to avoid overfitting)
   - **Random Forest Classifier** (Ensemble of decision trees, 100 estimators)
8. **Evaluation**:
   - Calculates **Accuracy**, **Precision**, **Recall**, and **F1-Score** on the test set.
   - Plots confusion matrices for each model.
   - Saves performance metrics in a bar chart to `model_comparison.png`.
9. **Predictions on Custom Cases**: Evaluates three manually defined applicants representing distinct risk levels using the best-performing model.

---

## Model Comparison Summary

*Typical performance results on the synthetic dataset:*

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8450** | **0.8333** | **0.6716** | **0.7438** |
| **Random Forest** | 0.8300 | 1.0000 | 0.4925 | 0.6600 |
| **Decision Tree** | 0.7650 | 0.7632 | 0.4328 | 0.5524 |

- **Logistic Regression** achieves the highest F1-Score (0.7438) and provides a balanced compromise between Precision and Recall.
- **Random Forest Classifier** exhibits exceptional Precision (1.0000), meaning it never approves a loan that should be rejected, though its Recall (0.4925) is lower (it rejects many loans that could have been approved).
- **Decision Tree** has a lower accuracy due to simple structure constraints.

---

## Sample Profiles Prediction Results

The best model (Logistic Regression) is tested on three profiles:
- **Case A: Ideal Applicant (High Credit Score & Income)** -> **APPROVED** (Confidence: 87.9%)
- **Case B: High Risk Applicant (Low Credit Score & Short Job History)** -> **REJECTED** (Confidence: 1.6%)
- **Case C: Borderline Applicant (Medium Credit & Medium Income)** -> **REJECTED** (Confidence: 36.2%)
