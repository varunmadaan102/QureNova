def qiskit_available():
    try:
        import qiskit  # noqa
        import qiskit_machine_learning  # noqa
        return True
    except Exception:
        return False

def build_feature_map(name="zz", num_qubits=4):
    if not qiskit_available():
        raise ImportError("Qiskit is not available.")
    from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
    name = name.lower()
    if name == "zz":
        return ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement="linear")
    if name == "z":
        return ZFeatureMap(feature_dimension=num_qubits, reps=2)
    raise ValueError(f"Unsupported feature map: {name}")
