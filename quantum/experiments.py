from config.settings import SUPPORTED_QUANTUM_QUBITS

def quantum_experiment_grid(feature_maps=("zz", "z"), qubits=None):
    qubits = qubits or SUPPORTED_QUANTUM_QUBITS
    return [{"feature_map": f, "num_qubits": q} for f in feature_maps for q in qubits]
