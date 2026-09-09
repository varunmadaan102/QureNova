"""
Navigation and page structure components for QureNova.

Provides consistent header treatment, breadcrumb navigation, and loading states
across all views to make the research workflow feel cohesive and legible.
"""
import streamlit as st
from components.theme import top_status


def page_header(title: str, description: str = None, icon: str = None):
    """
    Render a consistent page header with title, description, and optional icon.
    
    Replaces multiple manual calls to section_header and top_status.
    """
    top_status(title)
    icon_text = f"{icon} " if icon else ""
    
    st.markdown(f"## {icon_text}{title}")
    if description:
        st.markdown(f"*{description}*")
    st.divider()


def breadcrumb(items: list):
    """
    Render a breadcrumb navigation trail.
    
    Args:
        items: List of (label, key) tuples where key is the session state key.
               Last item is assumed to be current page (non-clickable).
    """
    if not items or len(items) < 2:
        return
    
    crumbs = []
    for i, (label, key) in enumerate(items):
        if i == len(items) - 1:
            # Current page - not clickable
            crumbs.append(f"**{label}**")
        else:
            # Previous pages - show as indicators
            crumbs.append(label)
    
    st.caption(" → ".join(crumbs))


def loading_spinner(message: str = "Processing..."):
    """Context manager for consistent loading state across all long operations."""
    return st.spinner(f"⏳ {message}")


def operation_status(success: bool, message: str, details: str = None):
    """
    Display operation result with consistent styling.
    
    Args:
        success: True for success, False for error/warning
        message: Primary message to display
        details: Optional detailed message (e.g., for expandable section)
    """
    if success:
        st.success(f"✓ {message}")
    else:
        st.error(f"✗ {message}")
    
    if details:
        with st.expander("Details"):
            st.code(details, language="text")


def workflow_checkpoint(step: int, total: int, label: str):
    """
    Render progress through a multi-step workflow.
    
    Args:
        step: Current step (1-indexed)
        total: Total steps in workflow
        label: Label for current step
    """
    progress = (step - 1) / total
    st.progress(progress)
    st.caption(f"Step {step}/{total} · {label}")


def data_status_panel():
    """
    Render consistent data availability status panel.
    
    Shows current dataset and experiment state, used as context reminder
    on views that depend on prior steps.
    """
    dataset = st.session_state.get("dataset")
    target = st.session_state.get("target_column")
    experiment = st.session_state.get("experiment_result")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if dataset is not None:
            st.metric("Dataset", f"{len(dataset)} rows", border=True)
        else:
            st.metric("Dataset", "None", border=True)
    
    with col2:
        if target is not None:
            st.metric("Target", target, border=True)
        else:
            st.metric("Target", "None", border=True)
    
    with col3:
        if experiment is not None:
            st.metric("Experiment", "Ready", border=True)
        else:
            st.metric("Experiment", "Pending", border=True)
