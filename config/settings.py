RANDOM_STATE = 42
CV_FOLDS = 5
DEFAULT_PCA_COMPONENTS = 4
DEFAULT_QUANTUM_QUBITS = 4
SUPPORTED_QUANTUM_QUBITS = [2, 4, 6]
# Fidelity-kernel evaluation is quadratic in the training sample count. Keep
# the interactive demonstration responsive while retaining a reproducible
# bounded subset of larger datasets.
MAX_QUANTUM_TRAIN_SAMPLES = 64
DEFAULT_TARGET_COLUMN = "diagnosis"
# Experimental review labels for the clinical decision-support surface. These
# are intentionally not presented as validated medical risk cutoffs.
RISK_THRESHOLDS = {
    "low_upper": 0.35,
    "moderate_upper": 0.65,
}

CLASSICAL_MODELS = [
    "Logistic Regression",
    "SVM (RBF)",
    "XGBoost",
]
