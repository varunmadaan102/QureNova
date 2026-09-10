# QureNova API boundary

The current application remains Streamlit-first. This directory is the future
HTTP/API boundary, not an active FastAPI deployment requirement.

The implementation must call domain/application services rather than importing
Streamlit views. `service.py` exposes the first framework-neutral entry point.
