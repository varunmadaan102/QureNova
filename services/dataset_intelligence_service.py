"""
Dataset intelligence service: Provide detailed profiling and preprocessing insights.
Answers: "What data did QureNova actually train on?"
"""

import numpy as np
import pandas as pd


def get_dataset_profile(df, experiment_result=None):
    """
    Generate a comprehensive profile of the dataset.
    
    Args:
        df: Input DataFrame
        experiment_result: Optional experiment result dict for preprocessing context
        
    Returns:
        dict with dataset profile information
    """
    profile = {
        "total_samples": len(df),
        "total_features": len(df.columns),
        "feature_names": df.columns.tolist(),
        "numeric_features": df.select_dtypes(include=[np.number]).columns.tolist(),
        "missing_values": {
            "count": int(df.isna().sum().sum()),
            "percentage": float((df.isna().sum().sum() / (len(df) * len(df.columns))) * 100),
            "by_column": {col: int(df[col].isna().sum()) for col in df.columns},
        },
        "duplicates": int(df.duplicated().sum()),
        "feature_statistics": {},
    }
    
    # Feature-level statistics
    for col in df.select_dtypes(include=[np.number]).columns:
        col_data = df[col].dropna()
        profile["feature_statistics"][col] = {
            "mean": float(col_data.mean()) if len(col_data) > 0 else None,
            "std": float(col_data.std()) if len(col_data) > 0 else None,
            "min": float(col_data.min()) if len(col_data) > 0 else None,
            "max": float(col_data.max()) if len(col_data) > 0 else None,
            "median": float(col_data.median()) if len(col_data) > 0 else None,
            "q25": float(col_data.quantile(0.25)) if len(col_data) > 0 else None,
            "q75": float(col_data.quantile(0.75)) if len(col_data) > 0 else None,
        }
    
    return profile


def get_preprocessing_context(experiment_result):
    """
    Extract preprocessing context from experiment result.
    
    Returns:
        dict describing what preprocessing was applied
    """
    preprocessing = experiment_result.get("preprocessing", {})
    dataset = experiment_result.get("dataset", {})
    metadata = experiment_result.get("metadata", {})
    
    context = {
        "dataset_rows": dataset.get("rows"),
        "dataset_features": dataset.get("features"),
        "feature_names": dataset.get("feature_names", []),
        "class_distribution": dataset.get("class_distribution", {}),
        "target_column": metadata.get("target_column"),
        "target_mapping": metadata.get("target_mapping", {}),
        "preprocessing_pipeline": preprocessing.get("pipeline", ""),
        "pca_components": preprocessing.get("pca_components"),
        "fit_scope": preprocessing.get("fit_scope", ""),
        "feature_selection": preprocessing.get("feature_selection", ""),
    }
    
    return context


def get_training_data_summary(experiment_result):
    """
    Summarize exactly what data was used for training.
    
    Returns:
        dict with training set composition for each model variant
    """
    classical = experiment_result.get("classical_results", {})
    classical_pca = experiment_result.get("classical_pca_results", {})
    quantum = experiment_result.get("quantum_results", {})
    
    dataset = experiment_result.get("dataset", {})
    preprocessing = experiment_result.get("preprocessing", {})
    
    total_rows = dataset.get("rows", 0)
    total_features = dataset.get("features", 0)
    
    summary = {
        "total_dataset_rows": total_rows,
        "total_dataset_features": total_features,
        "preprocessing_steps": preprocessing.get("pipeline", ""),
        "variants": {},
    }
    
    # Classical (no PCA)
    if classical:
        # With 5-fold CV, each fold uses 4/5 of data for training
        train_per_fold = int(total_rows * 0.8)
        summary["variants"]["classical_no_pca"] = {
            "description": "Classical models without dimensionality reduction",
            "validation_protocol": "Stratified 5-fold cross-validation",
            "samples_per_fold_train": train_per_fold,
            "samples_per_fold_test": total_rows - train_per_fold,
            "features_before_preprocessing": total_features,
            "features_after_preprocessing": total_features,
            "pca_applied": False,
            "note": "Each fold uses different train/test split; metrics are averages across folds",
        }
    
    # Classical + PCA
    if classical_pca:
        pca_components = preprocessing.get("pca_components")
        summary["variants"]["classical_pca"] = {
            "description": "Classical models with PCA control",
            "validation_protocol": "Stratified 5-fold cross-validation",
            "samples_per_fold_train": train_per_fold,
            "samples_per_fold_test": total_rows - train_per_fold,
            "features_before_preprocessing": total_features,
            "features_after_preprocessing": pca_components,
            "pca_applied": True,
            "pca_components": pca_components,
            "note": "PCA control isolates effect of dimensionality reduction from quantum effects",
        }
    
    # Quantum
    if quantum.get("available"):
        config = quantum.get("configuration", {})
        train_samples = config.get("train_samples", 0)
        test_samples = config.get("test_samples", 0) or (total_rows - train_samples)
        summary["variants"]["quantum_kernel"] = {
            "description": "Quantum kernel QSVC experiment",
            "validation_protocol": f"Holdout validation ({int(train_samples/(train_samples+test_samples)*100)}/{int(test_samples/(train_samples+test_samples)*100)} split)",
            "samples_train": train_samples,
            "samples_test": test_samples,
            "features_before_preprocessing": total_features,
            "features_after_preprocessing": config.get("num_qubits"),
            "pca_applied": True,
            "pca_components": config.get("num_qubits"),
            "feature_map": config.get("feature_map"),
            "note": "Quantum uses holdout validation (not CV) due to computational expense. Different protocol; not directly comparable by rank.",
        }
    
    return summary


def get_data_quality_warnings(df, experiment_result=None):
    """
    Generate warnings about potential data quality issues.
    
    Returns:
        list of warning messages
    """
    warnings = []
    
    # Check for duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        pct = (dup_count / len(df)) * 100
        warnings.append(f"⚠️ {dup_count} duplicate rows detected ({pct:.1f}% of dataset)")
    
    # Check for missing values
    missing_total = df.isna().sum().sum()
    if missing_total > 0:
        pct = (missing_total / (len(df) * len(df.columns))) * 100
        warnings.append(f"⚠️ {missing_total} missing values detected ({pct:.1f}% of data)")
        # Check which columns have the most missing
        missing_by_col = df.isna().sum()
        high_missing = missing_by_col[missing_by_col > len(df) * 0.1]
        if len(high_missing) > 0:
            cols = ", ".join(high_missing.index.tolist())
            warnings.append(f"   Columns with >10% missing: {cols}")
    
    # Check for zero variance features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    zero_variance = []
    for col in numeric_cols:
        if df[col].std() == 0:
            zero_variance.append(col)
    if zero_variance:
        warnings.append(f"⚠️ Zero-variance features detected: {', '.join(zero_variance[:5])}")
    
    # Check for extreme outliers (>5 sigma)
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            mean = col_data.mean()
            std = col_data.std()
            if std > 0:
                outliers = ((col_data - mean).abs() > 5 * std).sum()
                if outliers > 0:
                    pct = (outliers / len(col_data)) * 100
                    warnings.append(f"⚠️ {outliers} extreme outliers in '{col}' ({pct:.1f}% of values)")
    
    # Class imbalance warning (if target available)
    if experiment_result:
        dataset = experiment_result.get("dataset", {})
        class_dist = dataset.get("class_distribution", {})
        if class_dist:
            counts = list(class_dist.values())
            if len(counts) > 1:
                imbalance_ratio = max(counts) / min(counts)
                if imbalance_ratio > 3:
                    warnings.append(f"⚠️ Class imbalance detected (ratio {imbalance_ratio:.1f}:1)")
    
    return warnings


def get_feature_importance_context(experiment_result):
    """
    Extract feature context useful for explainability.
    
    Returns:
        dict with feature metadata for interpretation
    """
    dataset = experiment_result.get("dataset", {})
    preprocessing = experiment_result.get("preprocessing", {})
    
    return {
        "feature_names": dataset.get("feature_names", []),
        "feature_count": dataset.get("features", 0),
        "pca_components": preprocessing.get("pca_components"),
        "pca_applied_for_quantum": preprocessing.get("pca_components") is not None,
        "preprocessing_notes": preprocessing.get("pipeline", ""),
    }
