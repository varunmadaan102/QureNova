import argparse
import json
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", required=True, help="JSON encoded X (train+test).")
    parser.add_argument("--y_train", required=True)
    parser.add_argument("--y_test", required=True)
    parser.add_argument("--feature_map", default="zz")
    parser.add_argument("--num_qubits", type=int, default=2)
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    # Ensure repository root is on sys.path so `core.*` imports work when
    # invoked as a subprocess from different working directories.
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    # Note: we pass preprocessed numeric arrays already on purpose.
    X = np.asarray(json.loads(args.x), dtype=float)
    y_train = np.asarray(json.loads(args.y_train))
    y_test = np.asarray(json.loads(args.y_test))

    # Split point: worker assumes X is [X_train_q, X_test_q] concatenated along axis=0 with metadata.
    n_train = int(len(y_train))
    X_train_q = X[:n_train]
    X_test_q = X[n_train:]

    from core.evaluation import evaluate_predictions, build_confusion
    from quantum.alignment import kernel_target_alignment
    from quantum.kernel import compute_quantum_kernel, kernel_diagnostics
    from models.quantum import build_precomputed_qsvc

    train_kernel, test_kernel, circuit = compute_quantum_kernel(
        X_train_q,
        X_test_q,
        config={"feature_map": args.feature_map, "num_qubits": int(args.num_qubits)},
    )

    model = build_precomputed_qsvc()
    model.fit(train_kernel, y_train)
    pred = model.predict(test_kernel)
    score = model.decision_function(test_kernel)

    metrics = evaluate_predictions(y_true=y_test, y_pred=pred, y_score=score)
    alignment = kernel_target_alignment(train_kernel, y_train)

    try:
        circuit_text = str(circuit.draw(output="text"))
        if len(circuit_text) > 50000:
            circuit_text = None
            circuit_reason = "Circuit text rendering exceeded max length (50000 chars)."
        else:
            circuit_reason = None
    except Exception as e:
        circuit_text = None
        circuit_reason = f"Circuit text rendering failed: {type(e).__name__}: {e}"

    out = {
        "available": True,
        "metrics": metrics,
        "timing_seconds": None,
        "kernel_diagnostics": kernel_diagnostics(train_kernel),
        "kernel_target_alignment": float(alignment),
        "confusion_matrix": build_confusion(y_test, pred),
        "kernel_preview": train_kernel[:30, :30].tolist(),
        "circuit": circuit_text,
        "circuit_reason": circuit_reason,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
