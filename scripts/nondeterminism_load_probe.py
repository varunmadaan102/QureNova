"""Repeated experiment probes to detect nondeterministic failures / segfaults.

Runs combinations of:
- run_quantum: False/True
- cv_folds: 3/5
- quantum_qubits: 2/4/6
- feature_map: zz/z

Each (config, iteration) execution calls:
  services.experiment_service.run_experiment(df, config=...)

Results are saved as one JSON per config under the chosen output directory.

Usage:
  python scripts/nondeterminism_load_probe.py --iterations 3 --output_dir results/load_probe

No new dependencies are added; uses existing project modules.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data import load_demo_dataset
from services.experiment_service import run_experiment


@dataclass
class IterationResult:
    iteration_index: int
    ok: bool
    success: bool
    runtime_seconds: float
    output: Optional[Dict[str, Any]]
    error: Optional[str]


def _timestamped_filename(base: str) -> str:
    ts = int(time.time())
    return f"{base}_{ts}.json"


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def _make_config_slug(run_quantum: bool, cv_folds: int, quantum_qubits: int, feature_map: str) -> str:
    return (
        f"run_quantum={run_quantum}"
        f"_cv_folds={cv_folds}"
        f"_quantum_qubits={quantum_qubits}"
        f"_feature_map={feature_map}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Nondeterminism / segfault load probe")
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of repetitions per configuration.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/load_probe",
        help="Output directory for JSON result files.",
    )

    # Small-sample options: use config scaling while keeping requested option space.
    parser.add_argument(
        "--small_sample",
        action="store_true",
        help=(
            "If set, reduces the combination space to keep runs quick (existing run_quantum path respected)."
        ),
    )

    args = parser.parse_args()

    iterations = max(1, int(args.iterations))
    output_dir = Path(args.output_dir)

    # Option space. When --small_sample is set, reduce combos while still exercising quantum/non-quantum.
    if args.small_sample:
        run_quantum_values = [False, True]
        cv_folds_values = [3]
        quantum_qubits_values = [2]
        feature_map_values = ["zz"]
    else:
        run_quantum_values = [False, True]
        cv_folds_values = [3, 5]
        quantum_qubits_values = [2, 4, 6]
        feature_map_values = ["zz", "z"]

    df = load_demo_dataset()

    all_configs: List[Dict[str, Any]] = []
    for run_quantum in run_quantum_values:
        for cv_folds in cv_folds_values:
            for quantum_qubits in quantum_qubits_values:
                for feature_map in feature_map_values:
                    all_configs.append(
                        {
                            "run_quantum": bool(run_quantum),
                            "cv_folds": int(cv_folds),
                            "quantum_qubits": int(quantum_qubits),
                            "feature_map": str(feature_map),
                        }
                    )

    print(
        f"[load_probe] Starting: iterations={iterations}, configs={len(all_configs)}, output_dir={output_dir}"
    )
    sys.stdout.flush()

    for cfg in all_configs:
        slug = _make_config_slug(
            run_quantum=cfg["run_quantum"],
            cv_folds=cfg["cv_folds"],
            quantum_qubits=cfg["quantum_qubits"],
            feature_map=cfg["feature_map"],
        )

        run_dir = output_dir
        out_path = run_dir / _timestamped_filename(slug.replace("=", "").replace(" ", "_"))

        print(f"[load_probe] Config: {slug}")
        sys.stdout.flush()

        cfg_results: List[Dict[str, Any]] = []
        header = {
            "config": cfg,
            "iterations": iterations,
            "dataset": {
                "rows": int(len(df)),
            },
            "probe": {
                "script": "scripts/nondeterminism_load_probe.py",
                "root": str(ROOT),
            },
        }

        for i in range(iterations):
            started = time.perf_counter()
            ok = False
            success = False
            output = None
            err_text = None

            try:
                # NOTE: run_experiment returns a dict with nested classical/quantum results.
                # We store full output for later comparison.
                exp_config = {
                    "run_quantum": cfg["run_quantum"],
                    "cv_folds": cfg["cv_folds"],
                    "quantum_qubits": cfg["quantum_qubits"],
                    "feature_map": cfg["feature_map"],
                    # keep defaults for other knobs
                }

                output = run_experiment(df, config=exp_config)
                ok = True
                success = True
            except Exception as e:
                # Do not swallow BaseException subclasses (KeyboardInterrupt/SystemExit).
                # Those should propagate as failures.
                ok = False
                success = False
                err_text = (
                    f"{type(e).__name__}: {e}\n" + traceback.format_exc(limit=20)
                )
            except BaseException as e:  # noqa: B001
                # Explicitly do not swallow BaseException.
                raise
            finally:
                runtime_seconds = time.perf_counter() - started

            iter_result = IterationResult(
                iteration_index=i,
                ok=ok,
                success=success,
                runtime_seconds=float(runtime_seconds),
                output=output if ok else None,
                error=err_text,
            )
            cfg_results.append(asdict(iter_result))

            status_str = "SUCCESS" if success else "FAIL"
            print(
                f"[load_probe]  iter={i} status={status_str} runtime_seconds={runtime_seconds:.4f}"
            )
            if err_text:
                print(f"[load_probe]  error: {err_text.splitlines()[0]}")
            sys.stdout.flush()

        payload = {**header, "results": cfg_results}
        _json_dump(out_path, payload)
        print(f"[load_probe] Saved: {out_path}")
        sys.stdout.flush()

    print("[load_probe] Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
