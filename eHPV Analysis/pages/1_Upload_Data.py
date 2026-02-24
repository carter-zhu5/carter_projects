import pandas as pd
import streamlit as st
from utils.data import read_excel_safely

st.set_page_config(page_title="Upload Data", layout="wide")
st.title("Step 1 | Upload Data")

st.markdown(
    """
Upload an Excel (.xlsx) file with at least these columns (exact names):

- **Fahrzeug**
- **Modulgruppe**
- **Bauteil**
- **Aktuelle IEW eHPV**
- *(optional)* **Bauteil (Englisch)**
"""
)

uploaded = st.file_uploader("Upload your EHPV Excel file", type=["xlsx"])

REQUIRED_COLS = ["Fahrzeug", "Modulgruppe", "Bauteil", "Aktuelle IEW eHPV"]

if uploaded is not None:
    df, err = read_excel_safely(uploaded)
    if err:
        st.error(err)
        st.stop()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    # Normalize numeric
    df["Aktuelle IEW eHPV"] = pd.to_numeric(df["Aktuelle IEW eHPV"], errors="coerce").fillna(0)

    st.session_state["ehpv_df"] = df
    st.success("File uploaded and validated.")
    st.caption("Preview (first 15 rows):")
    st.dataframe(df.head(15), use_container_width=True)

    st.info("Open **Step 2 | Select Two Fahrzeug Models** from the sidebar.")

    # Show the button only if data is uploaded and validated
    if st.button("Go to Step 2 | Select Models"):
        st.switch_page("pages/2_Select_Models.py")
else:
    st.info("Please upload an Excel file to proceed.")