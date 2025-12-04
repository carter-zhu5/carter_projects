import streamlit as st
import pandas as pd

st.set_page_config(page_title="Select Models", layout="wide")
st.title("Step 2 | Select Two Models")

if "ehpv_df" not in st.session_state:
    st.error("No dataset found — please upload your file in Step 1.")
    st.stop()

df = st.session_state["ehpv_df"].copy()
if "Fahrzeug" not in df.columns:
    st.error("The uploaded file does not contain a 'Fahrzeug' column.")
    st.stop()

fahrzeug_list = sorted(pd.Series(df["Fahrzeug"]).dropna().unique().tolist())
if len(fahrzeug_list) < 2:
    st.warning("The dataset must include at least two different model values.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    m1 = st.selectbox("Predecessor:", [""] + fahrzeug_list, index=0, key="model1_select")
with col2:
    m2 = st.selectbox("Successor:", [""] + fahrzeug_list, index=0, key="model2_select")

if m1 and m2 and m1 == m2:
    st.warning("Please choose two different models.")
elif m1 and m2:
    st.success(f"Selected: {m1} vs {m2}")
    st.session_state["model1"] = m1
    st.session_state["model2"] = m2

    if st.button("Go to Step 3 | General Summary"):
        st.switch_page("pages/3_General_Summary.py")
else:
    st.info("Select two models to continue.")