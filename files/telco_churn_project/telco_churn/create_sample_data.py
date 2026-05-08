"""Create a sample Telco Churn dataset for demonstration."""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

# Sample size
n_samples = 7043

data = {
    'customerID': [f'6590-VHVEG{i:04d}' for i in range(n_samples)],
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'SeniorCitizen': np.random.choice([0, 1], n_samples),
    'Partner': np.random.choice(['Yes', 'No'], n_samples),
    'Dependents': np.random.choice(['Yes', 'No'], n_samples),
    'tenure': np.random.randint(0, 72, n_samples),
    'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
    'MonthlyCharges': np.random.uniform(20, 120, n_samples),
    'TotalCharges': np.random.uniform(100, 8000, n_samples),
    'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
}

df = pd.DataFrame(data)
df['MonthlyCharges'] = df['MonthlyCharges'].round(2)
df['TotalCharges'] = df['TotalCharges'].round(2)

output_path = Path(__file__).parent / 'data' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"[+] Sample dataset created: {output_path}")
print(f"    Shape: {df.shape}")
print(f"    Columns: {list(df.columns)}")
print(f"    Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
