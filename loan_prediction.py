import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Set premium visualization style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

def run_loan_prediction_pipeline():
    # 1. Load the dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "loan_data.csv")
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run generate_data.py first.")
        return
        
    print("=" * 80)
    print(" 1. LOADING DATASET ".center(80, "="))
    print("=" * 80)
    df = pd.read_csv(data_path)
    print(f"Dataset successfully loaded from: {data_path}")
    print(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    # 2. Explore the Data (EDA)
    print("=" * 80)
    print(" 2. EXPLORATORY DATA ANALYSIS (EDA) ".center(80, "="))
    print("=" * 80)
    print("\n--- First 5 Rows of the Dataset ---")
    print(df.head())
    
    print("\n--- Dataset Summary Statistics ---")
    print(df.describe(include='all'))
    
    # 3. Check for Missing Values & Impute
    print("\n" + "=" * 80)
    print(" 3. MISSING VALUES ANALYSIS & IMPUTATION ".center(80, "="))
    print("=" * 80)
    missing_count = df.isnull().sum()
    print("Missing values per column before imputation:")
    for col, val in missing_count.items():
        if val > 0:
            print(f" - {col}: {val} missing values ({val/len(df)*100:.1f}%)")
            
    # Copy DataFrame to avoid modifying raw data directly
    df_clean = df.copy()
    
    # Impute Numerical features using Median (robust to outliers)
    numerical_cols_with_nan = ['CreditScore', 'YearsAtJob']
    for col in numerical_cols_with_nan:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
        print(f" -> Imputed missing values in '{col}' with median: {median_val}")
        
    # Impute Categorical features using Mode (most frequent)
    categorical_cols_with_nan = ['Married', 'SelfEmployed']
    for col in categorical_cols_with_nan:
        mode_val = df_clean[col].mode()[0]
        df_clean[col] = df_clean[col].fillna(mode_val)
        print(f" -> Imputed missing values in '{col}' with mode: {mode_val}")
        
    print("\nMissing values after imputation:", df_clean.isnull().sum().sum())
    
    # 4. Convert Categorical columns to Numbers using LabelEncoder
    print("\n" + "=" * 80)
    print(" 4. CATEGORICAL VARIABLE ENCODING ".center(80, "="))
    print("=" * 80)
    
    categorical_cols = ['Gender', 'Married', 'Education', 'SelfEmployed']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        label_encoders[col] = le
        print(f" -> Encoded '{col}': {list(le.classes_)} -> {list(range(len(le.classes_)))}")
        
    # 5. Split Data into training (80%) and testing (20%)
    print("\n" + "=" * 80)
    print(" 5. DATA SPLITTING ".center(80, "="))
    print("=" * 80)
    
    # Features and Target
    X = df_clean.drop(columns=['ApplicantID', 'LoanApproved'])
    y = df_clean['LoanApproved']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set size:  X_test={X_test.shape}, y_test={y_test.shape}")
    
    # 6. Normalize Numerical Features using StandardScaler
    print("\n" + "=" * 80)
    print(" 6. FEATURE SCALING (NORMALIZATION) ".center(80, "="))
    print("=" * 80)
    
    numerical_cols = ['Age', 'AnnualIncome', 'CreditScore', 'LoanAmount', 'YearsAtJob']
    
    scaler = StandardScaler()
    # Fit scaler on training data and transform both train and test
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print(f"Normalized numerical columns: {numerical_cols}")
    print("Sample scaled training values:")
    print(X_train_scaled[numerical_cols].head(3))
    
    # 7. Train 3 models
    print("\n" + "=" * 80)
    print(" 7. MODEL TRAINING ".center(80, "="))
    print("=" * 80)
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=6)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Confusion Matrix": cm,
            "Predictions": y_pred
        }
        print(f" -> {name} trained successfully.")
        
    # 8. Compare models in a table
    print("\n" + "=" * 80)
    print(" 8. MODEL COMPARISON RESULTS ".center(80, "="))
    print("=" * 80)
    
    # Construct a clean DataFrame of results
    metrics_summary = []
    for name, metrics in results.items():
        metrics_summary.append({
            "Model": name,
            "Accuracy": f"{metrics['Accuracy']:.4f}",
            "Precision": f"{metrics['Precision']:.4f}",
            "Recall": f"{metrics['Recall']:.4f}",
            "F1-Score": f"{metrics['F1-Score']:.4f}"
        })
    summary_df = pd.DataFrame(metrics_summary)
    print(summary_df.to_string(index=False))
    
    # 9. Create Visualizations (Comparison Chart + Confusion Matrices)
    print("\n" + "=" * 80)
    print(" 9. CREATING VISUALIZATION PLOTS ".center(80, "="))
    print("=" * 80)
    
    # Make a grid of subplots: 1 top row for comparison, 3 bottom subplots for confusion matrices
    fig = plt.figure(figsize=(15, 12))
    
    # Add a custom aesthetic color palette
    colors = ['#4A90E2', '#50E3C2', '#F5A623']
    
    # Subplot 1: Metrics comparison
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    
    model_names = list(results.keys())
    metric_keys = ["Accuracy", "Precision", "Recall"]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    for i, metric in enumerate(metric_keys):
        values = [results[m][metric] for m in model_names]
        ax1.bar(x + i*width, values, width, label=metric, color=colors[i])
        
    ax1.set_title("Model Performance Comparison (Accuracy, Precision, Recall)", fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(model_names, fontsize=12)
    ax1.set_ylim(0.0, 1.1)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='none')
    
    # Add value labels above the bars
    for rects in ax1.containers:
        ax1.bar_label(rects, fmt='%.3f', padding=3, fontsize=9)
        
    # Subplots for Confusion Matrices
    for idx, (name, metrics) in enumerate(results.items()):
        ax = plt.subplot2grid((2, 3), (1, idx))
        cm = metrics["Confusion Matrix"]
        
        # Premium design with custom colormap
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Rejected', 'Approved'])
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False, values_format='d')
        
        ax.set_title(f"CM: {name}", fontsize=12, fontweight='bold', pad=10)
        ax.grid(False)
        
    plt.tight_layout()
    plot_path = os.path.join(current_dir, "model_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Evaluation plots saved to: {plot_path}")
    
    # Find the best model based on F1-Score (balances Precision and Recall)
    best_model_name = max(results.keys(), key=lambda m: results[m]["F1-Score"])
    best_model = trained_models[best_model_name]
    best_metrics = results[best_model_name]
    
    print(f"\n >>> BEST PERFORMING MODEL (by F1-score): {best_model_name} (F1: {best_metrics['F1-Score']:.4f}) <<<")
    
    # 10. Test the best model on custom sample cases
    print("\n" + "=" * 80)
    print(" 10. PREDICTING ON CUSTOM TEST CASES ".center(80, "="))
    print("=" * 80)
    
    # Define test samples (exactly matching columns of X)
    # X columns: ['Age', 'Gender', 'Married', 'Education', 'SelfEmployed', 'AnnualIncome', 'CreditScore', 'LoanAmount', 'YearsAtJob']
    
    # Encode values for sample cases using matching label encoders
    gender_m = label_encoders['Gender'].transform(['Male'])[0]
    gender_f = label_encoders['Gender'].transform(['Female'])[0]
    
    married_y = label_encoders['Married'].transform(['Yes'])[0]
    married_n = label_encoders['Married'].transform(['No'])[0]
    
    edu_g = label_encoders['Education'].transform(['Graduate'])[0]
    edu_ng = label_encoders['Education'].transform(['Not Graduate'])[0]
    
    se_y = label_encoders['SelfEmployed'].transform(['Yes'])[0]
    se_n = label_encoders['SelfEmployed'].transform(['No'])[0]

    samples = [
        {
            "Name": "Case A: Ideal Applicant (High Credit Score & Income)",
            "Data": pd.DataFrame([{
                "Age": 38, "Gender": gender_m, "Married": married_y, 
                "Education": edu_g, "SelfEmployed": se_n, "AnnualIncome": 120000.0, 
                "CreditScore": 780.0, "LoanAmount": 150000.0, "YearsAtJob": 8.0
            }])
        },
        {
            "Name": "Case B: High Risk Applicant (Low Credit Score & Short Job History)",
            "Data": pd.DataFrame([{
                "Age": 24, "Gender": gender_f, "Married": married_n, 
                "Education": edu_ng, "SelfEmployed": se_y, "AnnualIncome": 30000.0, 
                "CreditScore": 450.0, "LoanAmount": 120000.0, "YearsAtJob": 0.5
            }])
        },
        {
            "Name": "Case C: Borderline Applicant (Medium Credit & Medium Income)",
            "Data": pd.DataFrame([{
                "Age": 45, "Gender": gender_m, "Married": married_y, 
                "Education": edu_g, "SelfEmployed": se_n, "AnnualIncome": 65000.0, 
                "CreditScore": 640.0, "LoanAmount": 110000.0, "YearsAtJob": 4.5
            }])
        }
    ]
    
    for sample in samples:
        df_sample = sample["Data"]
        
        # Scale numerical fields using same scaler fit on training data
        df_sample_scaled = df_sample.copy()
        df_sample_scaled[numerical_cols] = scaler.transform(df_sample[numerical_cols])
        
        # Predict using best model
        prediction = best_model.predict(df_sample_scaled)[0]
        
        # Get probability if model supports it
        prob_str = ""
        if hasattr(best_model, "predict_proba"):
            prob = best_model.predict_proba(df_sample_scaled)[0]
            prob_approved = prob[1] * 100
            prob_str = f" (Confidence: {prob_approved:.1f}%)"
            
        status = "APPROVED" if prediction == 1 else "REJECTED"
        
        print(f"\n{sample['Name']}:")
        # Print readable input values
        orig = df_sample.iloc[0]
        print(f"   Input: Age={orig['Age']}, Income=${orig['AnnualIncome']:,.0f}, CreditScore={orig['CreditScore']:.0f}, LoanAmount=${orig['LoanAmount']:,.0f}, JobTenure={orig['YearsAtJob']} yrs")
        print(f"   Prediction: {status}{prob_str}")
        
    print("\n" + "=" * 80)
    print(" PIPELINE EXECUTION COMPLETED ".center(80, "="))
    print("=" * 80)

if __name__ == "__main__":
    run_loan_prediction_pipeline()
