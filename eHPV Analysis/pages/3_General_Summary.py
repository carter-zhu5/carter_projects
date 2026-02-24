import math
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openai import AzureOpenAI

# --- Streamlit page setup ---
st.set_page_config(page_title="EHPV Summary & Explorer", layout="wide")
st.title("Step 3 | EHPV Summary & Module Group Explorer")

# --- Azure OpenAI Client ---
def get_azure_client():
    """Initialize Azure OpenAI client from Streamlit secrets"""
    try:
        client = AzureOpenAI(
            azure_endpoint=st.secrets["azure_openai"]["endpoint"],
            api_key=st.secrets["azure_openai"]["api_key"],
            api_version="2024-02-01"
        )
        return client
    except Exception as e:
        st.error(f"Failed to initialize Azure OpenAI client: {e}")
        return None

# --- Short AI Analysis Function ---
def generate_short_analysis(m1: str, m2: str, part: str, delta_minutes: float, context: dict) -> str:
    """Generate very concise AI analysis"""
    client = get_azure_client()
    if not client:
        return "Analysis unavailable"
    
    prompt = f"""
    Explain this EHPV difference in ONE short sentence (max 15 words), focusing on Bauteilkonzept and Ausprägung:
    Part: {part}
    Models: {m1} vs {m2}
    Time difference: {delta_minutes:+.2f} minutes
    
    Note: Assembly process data may be outdated. Focus on design concepts.
    Be very concise.
    """
    
    try:
        response = client.chat.completions.create(
            model=st.secrets["azure_openai"]["deployment_name"],
            messages=[
                {
                    "role": "system", 
                    "content": "You are a manufacturing expert. Give extremely concise, one-sentence explanations focusing on Bauteilkonzept and Ausprägung. Maximum 15 words. Note that assembly data may be outdated."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        result = response.choices[0].message.content.strip()
        # Ensure it's short
        if len(result.split()) > 15:
            result = ' '.join(result.split()[:15]) + '...'
        return result
        
    except Exception as e:
        return "Analysis failed"

# --- Data processing function (moved to separate function) ---
def process_data():
    """Process data only when needed"""
    if "ehpv_df" not in st.session_state:
        st.error("No dataset found - please upload your file on the Home page.")
        st.stop()
    if "model1" not in st.session_state or "model2" not in st.session_state:
        st.error("Please select two Fahrzeug models in Step 2.")
        st.stop()

    df = st.session_state["ehpv_df"].copy()

    # --- Remove rows with Berichtstyp/SA == "SA" if present ---
    if "Berichtstyp/SA" in df.columns:
        df = df[df["Berichtstyp/SA"].astype(str).str.strip().str.upper() != "SA"]

    # --- Check for required columns ---
    required_cols = ["Fahrzeug", "Modulgruppe", "Bauteil", "Aktuelle IEW eHPV"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # --- Add English part name if missing ---
    if "Bauteil (Englisch)" not in df.columns:
        df["Bauteil (Englisch)"] = df["Bauteil"]

    # --- Model selection and validation ---
    m1 = st.session_state["model1"]
    m2 = st.session_state["model2"]

    # --- Filter data for selected models ---
    df1 = df[df["Fahrzeug"] == m1].copy()
    df2 = df[df["Fahrzeug"] == m2].copy()
    if df1.empty or df2.empty:
        st.error("One of the selected Fahrzeug models has no rows in the dataset.")
        st.stop()

    # --- Ensure numeric values for IEW columns ---
    df1["Aktuelle IEW eHPV"] = pd.to_numeric(df1["Aktuelle IEW eHPV"], errors="coerce").fillna(0.0)
    df2["Aktuelle IEW eHPV"] = pd.to_numeric(df2["Aktuelle IEW eHPV"], errors="coerce").fillna(0.0)

    # --- Merge model data for comparison ---
    value_cols = ["Modulgruppe", "Bauteil", "Bauteil (Englisch)", "Aktuelle IEW eHPV"]
    merged = pd.merge(
        df1[value_cols],
        df2[value_cols],
        on=["Modulgruppe", "Bauteil", "Bauteil (Englisch)"],
        how="outer",
        suffixes=(f"_{m1}", f"_{m2}"),
    )

    for col in (f"Aktuelle IEW eHPV_{m1}", f"Aktuelle IEW eHPV_{m2}"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    # --- Calculate delta between models ---
    delta_col = "Delta IEW eHPV"
    merged[delta_col] = merged[f"Aktuelle IEW eHPV_{m2}"] - merged[f"Aktuelle IEW eHPV_{m1}"]

    # --- Derive MG family from Modulgruppe ---
    def derive_mg_family(value: object) -> str:
        if pd.isna(value):
            return "Other"
        match = re.match(r"(\d+)", str(value))
        if not match:
            return "Other"
        first_digit = match.group(1)[0]
        return f"MG{first_digit}x" if first_digit in "12345" else "Other"

    merged["MG_Family"] = merged["Modulgruppe"].apply(derive_mg_family)

    # --- Calculate summary totals and percent change ---
    base_total = float(df1["Aktuelle IEW eHPV"].sum())
    total_model2 = float(df2["Aktuelle IEW eHPV"].sum())
    delta_total = total_model2 - base_total
    pct_change = (delta_total / base_total * 100.0) if base_total else 0.0

    return merged, m1, m2, delta_col, base_total, total_model2, delta_total, pct_change

# --- Main content with progress indication ---
with st.spinner("Processing data..."):
    merged, m1, m2, delta_col, base_total, total_model2, delta_total, pct_change = process_data()

# --- Display summary metrics ---
st.markdown("### Summary Totals")
c1, c2, c3 = st.columns(3)
c1.metric(f"{m1} total", f"{base_total:.1f} min")
c2.metric(f"{m2} total", f"{total_model2:.1f} min")
c3.metric("Delta total", f"{delta_total:+.1f} min", f"{pct_change:+.0f}%")
st.divider()

# --- Waterfall chart data preparation ---
with st.spinner("Generating waterfall chart..."):
    fam_summary = (
        merged.groupby("MG_Family", as_index=False)[delta_col]
        .sum()
        .sort_values("MG_Family")
    )

    x_vals = [m1] + fam_summary["MG_Family"].tolist() + [m2]
    y_vals = [base_total] + fam_summary[delta_col].tolist() + [total_model2]
    measures = ["absolute"] + ["relative"] * len(fam_summary) + ["total"]

    # --- Waterfall chart visualization ---
    fig = go.Figure(
        go.Waterfall(
            name="EHPV",
            orientation="v",
            measure=measures,
            x=x_vals,
            y=y_vals,
            text=[f"{v:.1f}" for v in y_vals],
            textposition="outside",
            connector={"line": {"color": "gray"}},
            increasing={"marker": {"color": "#d62728"}},
            decreasing={"marker": {"color": "#2ca02c"}},
            totals={"marker": {"color": "#4c78a8"}},
        )
    )
    fig.update_layout(
        title=f"Delta IEW eHPV - {m2} vs {m1} ({delta_total:+.1f} min / {pct_change:+.0f}%)",
        yaxis_title="EHPV [min]",
        height=540,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Module Group Explorer ---
st.markdown("---")
st.markdown("### Module Group Explorer")

# Get unique module groups and sort them
mg_families = sorted(merged["MG_Family"].dropna().unique().tolist())

# Create tabs for each module group
tabs = st.tabs(mg_families)

for i, mg in enumerate(mg_families):
    with tabs[i]:
        # Filter data for this MG family
        mg_data = merged[merged["MG_Family"] == mg].copy()
        
        # Skip if no data
        if mg_data.empty:
            st.info("No components found")
            continue
            
        # Calculate 80% impact for this MG family
        mg_total_delta = abs(mg_data[delta_col].sum())
        
        if mg_total_delta == 0:
            st.info("No time difference")
            continue
            
        mg_data["abs_delta"] = abs(mg_data[delta_col])
        mg_data = mg_data.sort_values("abs_delta", ascending=False)
        
        # Calculate cumulative percentage within this MG family
        mg_data["cumulative_pct"] = (mg_data["abs_delta"].cumsum() / mg_total_delta * 100)
        top_impact_mg = mg_data[mg_data["cumulative_pct"] <= 80]
        
        if top_impact_mg.empty:
            st.info("No significant components")
            continue
        
        # Display each top impact component
        for _, row in top_impact_mg.iterrows():
            col1, col2, col3 = st.columns([4, 2, 4])
            
            with col1:
                # Component name only
                st.write(row['Bauteil'])
            
            with col2:
                delta_val = row[delta_col]
                st.write(f"**{delta_val:+.1f} min**")
            
            with col3:
                if row[f"Aktuelle IEW eHPV_{m1}"] > 0 and row[f"Aktuelle IEW eHPV_{m2}"] > 0:
                    # Generate short AI analysis automatically
                    analysis_key = f"auto_{mg}_{row['Bauteil']}"
                    
                    if analysis_key not in st.session_state:
                        # Generate analysis automatically when first loading
                        context_payload = {
                            "part": row['Bauteil'],
                            "delta_minutes": delta_val,
                            "model_1": {"name": m1, "iew_ehpv": row[f"Aktuelle IEW eHPV_{m1}"]},
                            "model_2": {"name": m2, "iew_ehpv": row[f"Aktuelle IEW eHPV_{m2}"]}
                        }
                        
                        short_analysis = generate_short_analysis(
                            m1=m1, m2=m2, part=row['Bauteil'], 
                            delta_minutes=delta_val, context=context_payload
                        )
                        st.session_state[analysis_key] = short_analysis
                    
                    # Show the analysis result as plain text
                    if st.session_state[analysis_key]:
                        st.write(st.session_state[analysis_key])
                else:
                    # Show which model has this component
                    if row[f"Aktuelle IEW eHPV_{m1}"] > 0:
                        st.write(f"Only in {m1}")
                    else:
                        st.write(f"Only in {m2}")

# Navigation to detailed page
st.markdown("---")
if st.button("Open Detailed Analysis Page"):
    st.switch_page("pages/4_Detailed_Analysis.py")