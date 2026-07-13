import numpy as np
import pandas as pd
import os

def generate_loan_dataset(filepath, num_samples=1000, seed=42):
    np.random.seed(seed)
    
    # 1. Generate features
    applicant_ids = [f"LP{i:06d}" for i in range(1, num_samples + 1)]
    
    # Age: 21 to 70 years
    age = np.random.randint(21, 71, size=num_samples)
    
    # Annual Income: log-normal or skewed normal distribution between $20,000 and $250,000
    annual_income = np.round(np.random.exponential(scale=50000, size=num_samples) + 20000, -2)
    # Clip max income to $300,000
    annual_income = np.clip(annual_income, 20000, 300000)
    
    # Credit Score: 300 to 850, centered around 650 with standard deviation 100
    credit_score = np.random.normal(loc=650, scale=90, size=num_samples).astype(int)
    credit_score = np.clip(credit_score, 300, 850)
    
    # Loan Amount: generally correlates with income but has variance, say 0.5 to 5 times annual income
    income_multipliers = np.random.uniform(0.5, 4.0, size=num_samples)
    loan_amount = np.round(annual_income * income_multipliers, -2)
    # Clip loan amount from $5,000 to $600,000
    loan_amount = np.clip(loan_amount, 5000, 600000)
    
    # Years At Job: dependent on age but capped between 0 and 40
    # Average years at job increases with age
    years_at_job = np.zeros(num_samples)
    for i in range(num_samples):
        max_possible = min(40, age[i] - 18)
        if max_possible > 0:
            years_at_job[i] = np.round(np.random.uniform(0, max_possible), 1)
        else:
            years_at_job[i] = 0.0
            
    # Gender: 50% Male, 50% Female
    gender = np.random.choice(["Male", "Female"], size=num_samples, p=[0.5, 0.5])
    
    # Married: older people more likely to be married
    married = []
    for a in age:
        prob = 0.8 if a > 30 else 0.3
        married.append(np.random.choice(["Yes", "No"], p=[prob, 1 - prob]))
    married = np.array(married)
    
    # Education: 75% Graduate, 25% Not Graduate
    education = np.random.choice(["Graduate", "Not Graduate"], size=num_samples, p=[0.75, 0.25])
    
    # Self Employed: 15% Yes, 85% No
    self_employed = np.random.choice(["Yes", "No"], size=num_samples, p=[0.15, 0.85])
    
    # 2. Determine Loan Approval Target (with noise)
    # Define a logical risk score
    # High credit score: positive
    # Low debt-to-income ratio (LoanAmount / AnnualIncome): positive
    # Long employment: positive
    # Education = Graduate: positive
    
    # Let's normalize variables for the logit formula
    norm_credit = (credit_score - 300) / 550.0  # 0 to 1
    dti = loan_amount / annual_income           # debt-to-income ratio
    norm_income = (annual_income - 20000) / 280000.0
    norm_emp = years_at_job / 40.0
    
    # Logistic regression scoring function
    score = (
        -2.5 
        + 6.5 * norm_credit 
        - 1.8 * dti 
        + 1.5 * norm_income 
        + 1.2 * norm_emp 
        + 0.8 * (education == "Graduate").astype(int)
    )
    
    # Add random noise to make the dataset interesting and models testable
    noise = np.random.normal(loc=0, scale=0.8, size=num_samples)
    final_score = score + noise
    
    # Probability of approval
    prob = 1.0 / (1.0 + np.exp(-final_score))
    
    # Approved = 1, Rejected = 0
    loan_status = np.random.binomial(1, prob)
    
    # Create DataFrame
    df = pd.DataFrame({
        "ApplicantID": applicant_ids,
        "Age": age,
        "Gender": gender,
        "Married": married,
        "Education": education,
        "SelfEmployed": self_employed,
        "AnnualIncome": annual_income,
        "CreditScore": credit_score,
        "LoanAmount": loan_amount,
        "YearsAtJob": years_at_job,
        "LoanApproved": loan_status
    })
    
    # 3. Introduce missing values (NaNs) in specific columns to practice imputation
    # CreditScore: 5% missing
    cs_nan_indices = np.random.choice(num_samples, size=int(0.05 * num_samples), replace=False)
    df.loc[cs_nan_indices, "CreditScore"] = np.nan
    
    # YearsAtJob: 5% missing
    yj_nan_indices = np.random.choice(num_samples, size=int(0.05 * num_samples), replace=False)
    df.loc[yj_nan_indices, "YearsAtJob"] = np.nan
    
    # Married: 3% missing
    m_nan_indices = np.random.choice(num_samples, size=int(0.03 * num_samples), replace=False)
    df.loc[m_nan_indices, "Married"] = np.nan

    # SelfEmployed: 4% missing
    se_nan_indices = np.random.choice(num_samples, size=int(0.04 * num_samples), replace=False)
    df.loc[se_nan_indices, "SelfEmployed"] = np.nan

    # Save to CSV
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Dataset successfully generated with {num_samples} samples and saved to {filepath}")
    
if __name__ == "__main__":
    filepath = os.path.join(os.path.dirname(__file__), "loan_data.csv")
    generate_loan_dataset(filepath)
