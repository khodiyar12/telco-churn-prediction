from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
OUTPUT_DIR  = ROOT / "outputs"
MODEL_DIR   = ROOT / "models"

RAW_CSV     = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEAN_CSV   = DATA_DIR / "telco_clean.csv"

TARGET       = "Churn Value"
TARGET_LABEL = "Churn Label"
ID_COL       = "CustomerID"

DROP_COLS = [
    "CustomerID", "Count", "Country", "State", "City",
    "Zip Code", "Lat Long", "Latitude", "Longitude",
    "Churn Label", "Churn Score", "Churn Reason",
    "CLTV",
]

NUMERIC_COLS = [
    "Tenure Months", "Monthly Charges", "Total Charges",
]

BINARY_COLS = [
    "Gender", "Senior Citizen", "Partner", "Dependents",
    "Phone Service", "Paperless Billing",
]

MULTI_CAT_COLS = [
    "Multiple Lines", "Internet Service", "Online Security",
    "Online Backup", "Device Protection", "Tech Support",
    "Streaming TV", "Streaming Movies", "Contract", "Payment Method",
]

CATEGORICAL_COLS = BINARY_COLS + MULTI_CAT_COLS

RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5
SCORING      = "roc_auc"

LGBM_PARAM_SPACE = {
    "n_estimators":      (200, 1200),
    "learning_rate":     (0.01, 0.15),
    "num_leaves":        (20,  150),
    "max_depth":         (3,   10),
    "min_child_samples": (10,  80),
    "subsample":         (0.6, 1.0),
    "colsample_bytree":  (0.5, 1.0),
    "reg_alpha":         (1e-4, 10.0),
    "reg_lambda":        (1e-4, 10.0),
}
OPTUNA_TRIALS = 60
