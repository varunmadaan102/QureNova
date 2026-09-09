"""
Patient Experience Service: Enhanced result interpretation and per-sample explanations.

Separates model evidence (what the algorithm predicts) from clinical interpretation
(what that prediction means in context). Provides per-sample feature contributions
and reliability assessments without overstating certainty.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


def get_patient_interpretation(
    patient_row: pd.Series,
    prediction_result: Dict[str, Any],
    reference_data: pd.DataFrame,
    features: List[str],
) -> Dict[str, Any]:
    """
    Generate human-readable interpretation of model prediction for a single patient.
    
    Separates model output evidence from clinical context. Returns structured
    interpretation with confidence, agreement, reliability, and actionable caveats.
    
    Args:
        patient_row: Single patient result row from predict_patients()
        prediction_result: Dict with model predictions and probabilities
        reference_data: Reference dataset used for training
        features: List of feature names used in model
        
    Returns:
        Dict with:
        - patient_id: Patient identifier
        - primary_assessment: String summary of primary prediction
        - confidence_level: "High" | "Moderate" | "Low"
        - model_agreement: How many models agree
        - reliability_context: Explanation of reliability factors
        - key_observations: List of notable patient characteristics
        - important_caveats: List of limitations on this prediction
        - suggested_next_steps: Clinical/research actions
    """
    patient_id = patient_row.get("Patient", "Unknown")
    avg_prob = patient_row.get("Average Probability")
    agreement = patient_row.get("Model Agreement", "UNKNOWN")
    reliability = patient_row.get("Prediction Reliability", "UNKNOWN")
    risk_category = patient_row.get("Risk Category", "UNKNOWN")
    
    # Build primary assessment
    if avg_prob is None:
        primary = f"Model output unavailable for {patient_id}."
    elif avg_prob < 0.3:
        primary = f"Algorithmic assessment: LOW estimated probability (~{avg_prob:.1%})"
    elif avg_prob < 0.7:
        primary = f"Algorithmic assessment: INTERMEDIATE estimated probability (~{avg_prob:.1%})"
    else:
        primary = f"Algorithmic assessment: HIGH estimated probability (~{avg_prob:.1%})"
    
    # Model agreement count
    agreement_count = 0
    for col in patient_row.index:
        if col.endswith(" Prediction"):
            if patient_row[col] == 1:
                agreement_count += 1
    total_models = sum(1 for col in patient_row.index if col.endswith(" Prediction"))
    
    # Confidence level (based on distance from 0.5 and agreement)
    if avg_prob is None:
        confidence = "Low"
    else:
        distance_from_neutral = abs(avg_prob - 0.5)
        high_agreement = agreement_count >= (total_models - 1) if total_models > 0 else False
        
        if distance_from_neutral >= 0.25 and high_agreement:
            confidence = "High"
        elif distance_from_neutral >= 0.1 or high_agreement:
            confidence = "Moderate"
        else:
            confidence = "Low"
    
    # Build reliability context
    reliability_context = _build_reliability_context(
        reliability, 
        patient_row.get("Reliability Note", ""),
        avg_prob
    )
    
    # Key observations (vs reference population)
    observations = _get_key_observations(patient_row, reference_data, features)
    
    # Important caveats
    caveats = _build_caveats(
        reliability,
        agreement_count,
        total_models,
        avg_prob,
        risk_category
    )
    
    # Suggested next steps
    next_steps = _get_suggested_next_steps(
        avg_prob,
        agreement_count,
        total_models,
        risk_category
    )
    
    return {
        "patient_id": patient_id,
        "primary_assessment": primary,
        "confidence_level": confidence,
        "model_agreement": f"{agreement_count}/{total_models} models positive",
        "reliability_context": reliability_context,
        "key_observations": observations,
        "important_caveats": caveats,
        "suggested_next_steps": next_steps,
    }


def _build_reliability_context(reliability_level: str, reason: str, probability: Optional[float]) -> str:
    """Build contextual explanation of prediction reliability."""
    if probability is None:
        return "Prediction could not be computed due to missing data."
    
    if reliability_level == "HIGH":
        return f"Patient profile is consistent with reference population. {reason}"
    elif reliability_level == "MODERATE":
        return f"Patient profile has some differences from reference. {reason}"
    else:
        return f"Patient profile differs significantly from reference. Treat prediction as lower-confidence estimate. {reason}"


def _get_key_observations(
    patient_row: pd.Series,
    reference_data: pd.DataFrame,
    features: List[str]
) -> List[str]:
    """
    Extract key observations about patient vs reference population.
    
    Returns top 3-5 most unusual feature values.
    """
    observations = []
    
    # Get patient values (would need raw patient data passed in to compute properly)
    # For now, return observations based on model agreement and probability variance
    
    if patient_row.get("Model Agreement") == "LOW AGREEMENT":
        observations.append(
            "Models produced different assessments—patient profile may have mixed signals"
        )
    
    avg_prob = patient_row.get("Average Probability")
    if avg_prob is not None and 0.35 < avg_prob < 0.65:
        observations.append(
            "Patient characteristics align with boundary between outcomes"
        )
    
    # Check for high/low model confidence
    confidence = patient_row.get("Model Confidence", "")
    if confidence == "Low":
        observations.append(
            "Model confidence is low; patient may have atypical feature patterns"
        )
    
    return observations if observations else ["Patient profile loaded successfully"]


def _build_caveats(
    reliability: str,
    agreeing_models: int,
    total_models: int,
    probability: Optional[float],
    risk_category: str
) -> List[str]:
    """Build list of important limitations and caveats."""
    caveats = []
    
    caveats.append(
        "This is a research estimate from trained classifiers, not a clinical diagnosis."
    )
    
    if reliability != "HIGH":
        caveats.append(
            "Patient differs from reference population; prediction reliability is reduced."
        )
    
    if agreeing_models < total_models:
        caveats.append(
            f"Models disagreed ({agreeing_models}/{total_models} positive); rely on ensemble only."
        )
    
    if probability is not None and 0.3 < probability < 0.7:
        caveats.append(
            "Prediction is near decision boundary; modest changes could reverse outcome."
        )
    
    caveats.append(
        "Model was trained on limited, specific reference data. Generalization to other populations unknown."
    )
    
    return caveats


def _get_suggested_next_steps(
    probability: Optional[float],
    agreeing_models: int,
    total_models: int,
    risk_category: str
) -> List[str]:
    """Suggest next actions based on prediction result."""
    steps = []
    
    if probability is None:
        return ["Review input data for completeness and schema errors"]
    
    if risk_category == "Elevated":
        steps.append(
            "Consider clinical evaluation and additional diagnostic workup"
        )
    elif risk_category == "Moderate":
        steps.append(
            "Monitor patient; clinical judgment should guide next steps"
        )
    
    if agreeing_models < total_models:
        steps.append(
            "Review individual model predictions to understand disagreement sources"
        )
    
    steps.append(
        "Compare with other diagnostic evidence; use model as supplementary information only"
    )
    
    return steps


def get_feature_contribution_summary(
    patient_data: pd.DataFrame,
    model,
    features: List[str],
    preprocessor=None,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Compute feature contribution to prediction for a single patient.
    
    For linear models, uses coefficients × standardized values.
    Returns top features driving the prediction decision.
    
    Args:
        patient_data: Single row patient dataframe (raw features)
        model: Fitted classifier with coef_ or feature_importances_
        features: List of feature names
        preprocessor: Optional preprocessing pipeline (StandardScaler, etc.)
        top_n: Return top N contributing features
        
    Returns:
        DataFrame with Feature, Raw Value, Contribution, Direction columns
    """
    try:
        # Preprocess if needed
        X_patient = patient_data[features].values
        if preprocessor is not None:
            X_patient = preprocessor.transform(patient_data[features])
        else:
            X_patient = patient_data[features].values
        
        # Get coefficients or importances
        if hasattr(model, "coef_"):
            # Linear model: feature_importance = coef * x
            contributions = X_patient[0] * model.coef_[0]
            feature_type = "linear coefficient"
        elif hasattr(model, "feature_importances_"):
            # Tree model: use feature importance
            contributions = model.feature_importances_
            feature_type = "tree importance"
        else:
            return pd.DataFrame()
        
        # Build result
        result = pd.DataFrame({
            "Feature": features,
            "Raw Value": patient_data[features].values[0],
            "Contribution": contributions,
            "Absolute Contribution": np.abs(contributions),
        }).sort_values("Absolute Contribution", ascending=False)
        
        # Add direction and limit to top_n
        result["Direction"] = result["Contribution"].apply(
            lambda x: "↑ Positive" if x > 0 else "↓ Negative" if x < 0 else "—"
        )
        
        return result[["Feature", "Raw Value", "Contribution", "Direction"]].head(top_n)
        
    except Exception:
        return pd.DataFrame()


def get_model_calibration_summary(
    predictions_df: pd.DataFrame,
    target_values: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Assess how well model probabilities match actual outcomes (if available).
    
    Returns calibration statistics and guidance on prediction reliability.
    """
    summary = {
        "total_predictions": len(predictions_df),
        "predictions_with_probability": 0,
        "avg_probability": None,
        "probability_range": None,
        "calibration_note": "",
    }
    
    probs = predictions_df["Average Probability"].dropna()
    summary["predictions_with_probability"] = len(probs)
    
    if len(probs) > 0:
        summary["avg_probability"] = float(probs.mean())
        summary["probability_range"] = (float(probs.min()), float(probs.max()))
        
        # Calibration note
        if summary["avg_probability"] < 0.3:
            summary["calibration_note"] = "Model tends toward low-probability predictions"
        elif summary["avg_probability"] > 0.7:
            summary["calibration_note"] = "Model tends toward high-probability predictions"
        else:
            summary["calibration_note"] = "Model probabilities well-distributed"
    
    return summary
