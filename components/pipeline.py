import streamlit as st

def pipeline(items):
    st.markdown(
        '<div class="q-pipeline">'
        + " <span aria-hidden='true'>→</span> ".join(
            [f"<strong>{x}</strong>" for x in items]
        )
        + "</div>",
        unsafe_allow_html=True,
    )
