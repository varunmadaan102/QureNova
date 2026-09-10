"""
Error handling and user feedback components for QureNova.

Provides consistent, clear error messages and recovery guidance
that help users understand what went wrong and how to proceed.
"""
import streamlit as st
from typing import Optional, List


class ValidationError:
    """Structured validation error with guidance."""
    
    def __init__(self, 
                 title: str, 
                 message: str,
                 suggestions: List[str] = None,
                 technical_details: str = None):
        self.title = title
        self.message = message
        self.suggestions = suggestions or []
        self.technical_details = technical_details
    
    def display(self):
        """Render the error in a user-friendly format."""
        st.error(f"**{self.title}**\n\n{self.message}")
        
        if self.suggestions:
            st.info("**What you can try:**")
            for i, suggestion in enumerate(self.suggestions, 1):
                st.write(f"{i}. {suggestion}")
        
        if self.technical_details:
            with st.expander("Technical details"):
                st.code(self.technical_details, language="text")


class InputWarning:
    """Structured warning about input quality or edge cases."""
    
    def __init__(self,
                 title: str,
                 message: str,
                 impact: str = None):
        self.title = title
        self.message = message
        self.impact = impact
    
    def display(self):
        """Render the warning in a user-friendly format."""
        st.warning(f"**{self.title}**\n\n{self.message}")
        if self.impact:
            st.caption(f"**Impact:** {self.impact}")


def handle_schema_error(error: Exception, context: str = "data") -> ValidationError:
    """Convert schema validation errors into user guidance."""
    error_msg = str(error).lower()
    
    if "missing" in error_msg and "features" in error_msg:
        return ValidationError(
            title="Missing required features",
            message="The uploaded CSV is missing one or more required diagnostic feature columns.",
            suggestions=[
                "Check that all 30 feature columns are present in your CSV",
                "Verify column names match the expected format (e.g., mean_radius, radius_mean, radius_se, radius_worst)",
                "Download the template CSV from Data Workspace and compare your headers",
            ],
            technical_details=str(error)
        )
    
    elif "duplicate" in error_msg:
        return ValidationError(
            title="Duplicate column names",
            message="Your CSV has duplicate or ambiguous column headers.",
            suggestions=[
                "Remove duplicate columns",
                "Ensure each feature name appears exactly once",
                "Check for whitespace or case differences in column names",
            ],
            technical_details=str(error)
        )
    
    elif "non-numeric" in error_msg or "numeric" in error_msg:
        return ValidationError(
            title="Non-numeric feature values",
            message="One or more feature columns contain non-numeric values (text, symbols, etc.).",
            suggestions=[
                "Remove rows with text in numeric columns",
                "Replace empty cells with empty (leave blank for imputation)",
                "Check for embedded units like '12 mm' instead of '12'",
            ],
            technical_details=str(error)
        )
    
    elif "target" in error_msg and "binary" in error_msg:
        return ValidationError(
            title="Target is not binary",
            message="The selected target column must have exactly 2 unique values (binary classification).",
            suggestions=[
                "Choose a different target column with exactly 2 classes",
                "If your target has 3+ classes, consider selecting one vs. rest",
                "Ensure no missing values in the target column",
            ],
            technical_details=str(error)
        )
    
    elif "missing" in error_msg and "target" in error_msg:
        return ValidationError(
            title="Missing target values",
            message="The target column contains missing (blank/null) values.",
            suggestions=[
                "Remove rows where target is missing",
                "Select a different target column with complete values",
            ],
            technical_details=str(error)
        )
    
    else:
        return ValidationError(
            title=f"Schema validation error",
            message="The uploaded CSV does not match the expected structure.",
            suggestions=[
                "Review the CSV format requirements in Data Workspace",
                "Download and use the template CSV provided",
                "Check that your file is properly formatted as comma-separated values",
            ],
            technical_details=str(error)
        )


def handle_computation_error(error: Exception, operation: str = "operation") -> ValidationError:
    """Convert computation errors into user guidance."""
    error_msg = str(error).lower()
    
    if "memory" in error_msg or "out of" in error_msg:
        return ValidationError(
            title=f"{operation.title()} exceeded memory limits",
            message="The dataset or model configuration is too large for this operation.",
            suggestions=[
                "Reduce the dataset size (fewer rows)",
                "Use fewer features if possible",
                "Close other applications to free up memory",
                "Try again later when system load is lower",
            ],
            technical_details=str(error)
        )
    
    elif "timeout" in error_msg or "time" in error_msg:
        return ValidationError(
            title=f"{operation.title()} timed out",
            message="The operation took too long to complete.",
            suggestions=[
                "Try with a smaller dataset",
                "Reduce the number of features",
                "Try again—network or system load may have improved",
            ],
            technical_details=str(error)
        )
    
    elif "quantum" in error_msg.lower() or "qiskit" in error_msg.lower():
        return ValidationError(
            title="Quantum backend error",
            message="An error occurred during quantum simulation.",
            suggestions=[
                "Verify Qiskit and dependencies are installed",
                "Reduce dataset size for smaller circuit depth",
                "Check system resources (CPU, memory)",
                "Review quantum backend configuration",
            ],
            technical_details=str(error)
        )
    
    else:
        return ValidationError(
            title=f"{operation.title()} failed",
            message="An unexpected error occurred during processing.",
            suggestions=[
                "Check your input data for completeness",
                "Try with a smaller dataset",
                "Review system resources",
                "Contact support with the technical details below",
            ],
            technical_details=str(error)
        )


def warn_about_dataset_quality(issues: List[str]):
    """Display warnings about dataset quality issues."""
    if not issues:
        return
    
    st.warning("**Data quality observations**")
    for issue in issues:
        st.write(f"• {issue}")


def warn_about_prediction_reliability(reliability: str, reason: str):
    """Display specific warning about prediction reliability."""
    if reliability == "HIGH":
        st.success("✓ High confidence prediction—patient profile aligns with reference population")
    elif reliability == "MODERATE":
        st.warning(f"⚠ Moderate confidence—{reason}")
    else:
        st.error(f"✗ Low confidence—{reason}. Treat estimate as exploratory only.")


def suggest_workflow_next_steps(current_page: str) -> List[str]:
    """Suggest next steps based on current page in workflow."""
    workflow = {
        "overview": [
            "→ Start in Data Workspace to load or generate a dataset",
            "→ Choose a binary target column",
        ],
        "data_workspace": [
            "→ Review dataset schema and quality metrics",
            "→ Proceed to Experiment Lab when ready",
        ],
        "experiment_lab": [
            "→ Configure experiment parameters",
            "→ Run classical + quantum comparison",
            "→ View results in Benchmark",
        ],
        "benchmark": [
            "→ Compare model performance across protocols",
            "→ Review PCA effect analysis",
            "→ Use Patient Analysis for individual predictions",
        ],
        "patient_analysis": [
            "→ Review per-sample predictions and confidence",
            "→ Check explainability for feature contributions",
        ],
    }
    return workflow.get(current_page, [])
