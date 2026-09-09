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

CLASSICAL_MODELS = [
    "Logistic Regression",
    "SVM (RBF)",
    "XGBoost",
]
