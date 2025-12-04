import streamlit as st

st.set_page_config(page_title="eHPV Waterfall Analysis Compare", layout="wide")
st.title("eHPV Waterfall Analysis App")

st.markdown(
    """
Welcome. Use the sidebar to open each step:

1) **Upload Data** – load your Excel with eHPV.
2) **Select Models** – pick two *Fahrzeug* to compare.
3) **Compare EHPV** – waterfall + detail + (optional) AI summary.

> Tip (Databricks): after opening this file in the workspace, choose **Run as Streamlit app**.
"""
)

with st.expander("What do I need to prepare?"):
    st.markdown(
        """
- Excel `.xlsx` with at least: **Fahrzeug, Modulgruppe, Bauteil, Aktuelle IEW eHPV**  
        """
    )
