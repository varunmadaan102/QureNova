"""
Comparison service: Merge classical, classical+PCA, and quantum results
into a unified, fair, protocol-transparent benchmark.
"""

import pandas as pd
import numpy as np


def merge_all_results(experiment_result):
    """
    Merge classical, classical_pca, and quantum results into a single
    comparison DataFrame with protocol labels and fair comparison warnings.
    
    Args:
        experiment_result: Full experiment result dict from experiment_service
        
    Returns:
        DataFrame with columns: Model, Protocol, Samples, Features, Accuracy, 
        BalancedAccuracy, Sensitivity, Specificity, Precision, F1, ROC_AUC, 
        Time(s), Note
    """
    rows = []
    
    # Extract metadata
    dataset = experiment_result.get("dataset", {})
    preprocessing = experiment_result.get("preprocessing", {})
    pca_components = preprocessing.get("pca_components")
    
    # Classical results (no PCA)
    classical_results = experiment_result.get("classical_results", {})
    for model_name, model_data in classical_results.items():
        summary = model_data.get("summary", {})
        rows.append({
            "Model": model_name,
            "Variant": "Classical (no PCA)",
            "Protocol": "Stratified 5-fold CV",
            "Samples": "all",
            "Features": "all",
            "Accuracy": summary.get("accuracy", {}).get("mean"),
            "BalancedAccuracy": summary.get("balanced_accuracy", {}).get("mean"),
            "Sensitivity": summary.get("sensitivity", {}).get("mean"),
            "Specificity": summary.get("specificity", {}).get("mean"),
            "Precision": summary.get("precision", {}).get("mean"),
            "F1": summary.get("f1", {}).get("mean"),
            "ROC_AUC": summary.get("roc_auc", {}).get("mean"),
            "Time(s)": model_data.get("timing_seconds", 0),
            "Note": "Baseline without dimensionality reduction",
        })
    
    # Classical+PCA results (control)
    classical_pca_results = experiment_result.get("classical_pca_results", {})
    if classical_pca_results:
        for model_name, model_data in classical_pca_results.items():
            summary = model_data.get("summary", {})
            pca_label = f"PCA ({pca_components} components)" if pca_components else "PCA"
            rows.append({
                "Model": model_name,
                "Variant": "Classical (with PCA)",
                "Protocol": "Stratified 5-fold CV",
                "Samples": "all",
                "Features": f"reduced to {pca_components}" if pca_components else "reduced",
                "Accuracy": summary.get("accuracy", {}).get("mean"),
                "BalancedAccuracy": summary.get("balanced_accuracy", {}).get("mean"),
                "Sensitivity": summary.get("sensitivity", {}).get("mean"),
                "Specificity": summary.get("specificity", {}).get("mean"),
                "Precision": summary.get("precision", {}).get("mean"),
                "F1": summary.get("f1", {}).get("mean"),
                "ROC_AUC": summary.get("roc_auc", {}).get("mean"),
                "Time(s)": model_data.get("timing_seconds", 0),
                "Note": f"PCA control: measures effect of {pca_label}",
            })
    
    # Quantum results (if available)
    quantum_results = experiment_result.get("quantum_results", {})
    if quantum_results.get("available"):
        metrics = quantum_results.get("metrics", {})
        config = quantum_results.get("configuration", {})
        feature_map = config.get("feature_map", "zz").upper()
        num_qubits = config.get("num_qubits", "?")
        model_name = f'QSVC ({feature_map}-{num_qubits}Q)'
        rows.append({
            "Model": model_name,
            "Variant": "Quantum Kernel",
            "Protocol": "Holdout validation (80/20 split)",
            "Samples": f"~{config.get('training_samples', '?')} (holdout)",
            "Features": f"reduced to {config.get('num_qubits', '?')}",
            "Accuracy": metrics.get("accuracy"),
            "BalancedAccuracy": metrics.get("balanced_accuracy"),
            "Sensitivity": metrics.get("sensitivity"),
            "Specificity": metrics.get("specificity"),
            "Precision": metrics.get("precision"),
            "F1": metrics.get("f1"),
            "ROC_AUC": metrics.get("roc_auc"),
            "Time(s)": quantum_results.get("timing_seconds", 0),
            "Note": "Simulation-based; holdout protocol differs from CV",
        })
    
    return pd.DataFrame(rows)


def get_comparison_narrative(experiment_result):
    """
    Generate a fair, honest narrative about the comparison results.
    
    Returns:
        dict with keys: title, summary, protocol_warning, quantum_assessment
    """
    classical_results = experiment_result.get("classical_results", {})
    classical_pca_results = experiment_result.get("classical_pca_results", {})
    quantum_results = experiment_result.get("quantum_results", {})
    
    # Find best classical performance
    best_classical = None
    best_acc = -1
    for model_name, model_data in classical_results.items():
        acc = model_data.get("summary", {}).get("accuracy", {}).get("mean", 0)
        if acc > best_acc:
            best_acc = acc
            best_classical = model_name
    
    # Find quantum performance
    quantum_acc = None
    if quantum_results.get("available"):
        quantum_acc = quantum_results.get("metrics", {}).get("accuracy")
    
    # Build narrative
    title = "Fair Benchmarking: Transparent Protocol Comparison"
    
    summary = f"Classical baseline ({best_classical}): {best_acc:.1%} accuracy. "
    
    if classical_pca_results:
        best_pca = None
        best_pca_acc = -1
        for model_name, model_data in classical_pca_results.items():
            acc = model_data.get("summary", {}).get("accuracy", {}).get("mean", 0)
            if acc > best_pca_acc:
                best_pca_acc = acc
                best_pca = model_name
        pca_delta = best_pca_acc - best_acc
        summary += f"Classical+PCA ({best_pca}): {best_pca_acc:.1%} accuracy ({pca_delta:+.1%} vs baseline). "
    
    if quantum_acc:
        quantum_delta = quantum_acc - best_acc
        summary += f"Quantum kernel: {quantum_acc:.1%} accuracy ({quantum_delta:+.1%} vs classical baseline). "
        
        if quantum_delta < 0:
            assessment = (
                "In this experiment, classical baselines outperformed the evaluated quantum kernel configuration. "
                "This result demonstrates why rigorous benchmarking protocols are essential before claiming quantum advantage. "
                "Quantum may excel in different feature spaces, dataset sizes, or with optimized hyperparameters."
            )
        else:
            assessment = (
                "Quantum kernel achieved competitive performance with classical baselines. "
                "However, this is a single simulation-based experiment; results must be validated across multiple datasets and configurations."
            )
    else:
        assessment = "Quantum experiment was not run in this trial."
    
    protocol_warning = (
        "⚠️ IMPORTANT: Classical models used stratified 5-fold cross-validation; "
        "quantum used holdout validation (80/20 split). These are different evaluation protocols. "
        "Results are NOT directly comparable due to different sample sizes and validation methods. "
        "For a true apples-to-apples comparison, both should use identical train/test splits."
    )
    
    return {
        "title": title,
        "summary": summary,
        "protocol_warning": protocol_warning,
        "quantum_assessment": assessment,
    }


def get_pca_effect_analysis(experiment_result):
    """
    Analyze the effect of PCA on classical model performance.
    
    Returns:
        dict with analysis of accuracy/F1 changes from adding PCA
    """
    classical_results = experiment_result.get("classical_results", {})
    classical_pca_results = experiment_result.get("classical_pca_results", {})
    
    if not classical_pca_results:
        return None
    
    analysis = []
    for model_name in classical_results:
        if model_name not in classical_pca_results:
            continue
            
        baseline_acc = classical_results[model_name].get("summary", {}).get("accuracy", {}).get("mean", 0)
        pca_acc = classical_pca_results[model_name].get("summary", {}).get("accuracy", {}).get("mean", 0)
        
        baseline_f1 = classical_results[model_name].get("summary", {}).get("f1", {}).get("mean", 0)
        pca_f1 = classical_pca_results[model_name].get("summary", {}).get("f1", {}).get("mean", 0)
        
        analysis.append({
            "model": model_name,
            "accuracy_baseline": baseline_acc,
            "accuracy_pca": pca_acc,
            "accuracy_delta": pca_acc - baseline_acc,
            "f1_baseline": baseline_f1,
            "f1_pca": pca_f1,
            "f1_delta": pca_f1 - baseline_f1,
        })
    
    return analysis


def get_quantum_vs_classical_summary(experiment_result):
    """
    Generate a summary comparing quantum performance to best classical baseline.
    
    Returns:
        dict with detailed comparison metrics
    """
    classical_results = experiment_result.get("classical_results", {})
    quantum_results = experiment_result.get("quantum_results", {})
    
    if not quantum_results.get("available"):
        return {"status": "not_run"}
    
    # Find best classical
    best_classical_name = None
    best_classical_metrics = None
    best_classical_acc = -1
    
    for model_name, model_data in classical_results.items():
        acc = model_data.get("summary", {}).get("accuracy", {}).get("mean", 0)
        if acc > best_classical_acc:
            best_classical_acc = acc
            best_classical_name = model_name
            best_classical_metrics = model_data.get("summary", {})
    
    quantum_metrics = quantum_results.get("metrics", {})
    
    return {
        "status": "completed",
        "quantum_vs_best_classical": {
            "classical_model": best_classical_name,
            "classical_accuracy": best_classical_metrics.get("accuracy", {}).get("mean", 0) if best_classical_metrics else 0,
            "quantum_accuracy": quantum_metrics.get("accuracy", 0),
            "accuracy_delta": quantum_metrics.get("accuracy", 0) - (best_classical_metrics.get("accuracy", {}).get("mean", 0) if best_classical_metrics else 0),
            "classical_balanced_accuracy": best_classical_metrics.get("balanced_accuracy", {}).get("mean", 0) if best_classical_metrics else 0,
            "quantum_balanced_accuracy": quantum_metrics.get("balanced_accuracy", 0),
            "classical_f1": best_classical_metrics.get("f1", {}).get("mean", 0) if best_classical_metrics else 0,
            "quantum_f1": quantum_metrics.get("f1", 0),
            "classical_roc_auc": best_classical_metrics.get("roc_auc", {}).get("mean", 0) if best_classical_metrics else 0,
            "quantum_roc_auc": quantum_metrics.get("roc_auc", 0),
        },
        "quantum_metadata": {
            "feature_map": quantum_results.get("configuration", {}).get("feature_map"),
            "num_qubits": quantum_results.get("configuration", {}).get("num_qubits"),
            "training_samples": quantum_results.get("configuration", {}).get("training_samples"),
            "kernel_computation_time": quantum_results.get("timing_seconds", 0),
            "kernel_alignment": quantum_results.get("metrics", {}).get("kernel_target_alignment"),
        },
    }
