import numpy as np
from quantum.feature_maps import build_feature_map, qiskit_available

def compute_quantum_kernel(X_train, X_test=None, config=None):
    config = config or {}
    if not qiskit_available():
        raise ImportError("Qiskit/qiskit-machine-learning is not installed.")

    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    feature_map = build_feature_map(
        config.get("feature_map", "zz"),
        config.get("num_qubits", X_train.shape[1]),
    )
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    train_kernel = kernel.evaluate(X_train)
    test_kernel = kernel.evaluate(X_test, X_train) if X_test is not None else None

    return train_kernel, test_kernel, feature_map

def kernel_diagnostics(matrix):
    m = np.asarray(matrix, dtype=float)
    return {
        "shape": list(m.shape),
        "min": float(np.min(m)),
        "max": float(np.max(m)),
        "mean": float(np.mean(m)),
        "std": float(np.std(m)),
    }
