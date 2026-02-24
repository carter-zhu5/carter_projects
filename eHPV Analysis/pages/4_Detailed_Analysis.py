import math
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from openai import AzureOpenAI

# --- Streamlit page setup ---
st.set_page_config(page_title="Detailed EHPV Analysis", layout="wide")
st.title("Step 4 | Detailed EHPV Analysis")

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

# --- Full AI Analysis Function ---
def generate_ai_analysis(m1: str, m2: str, part: str, modulgruppe: str, 
                        delta_minutes: float, language: str, context: dict) -> str:
    """Generate detailed AI analysis"""
    client = get_azure_client()
    if not client:
        return "Error: Azure OpenAI client not available."
    
    model1_data = context.get("model_1", {})
    model2_data = context.get("model_2", {})
    
    prompt_parts = [
        f"You are a manufacturing engineering expert specializing in automotive assembly processes and time analysis.",
        f"Explain why Δ IEW eHPV = {delta_minutes:+.2f} minutes between {m1} and {m2} for part: {part}",
        f"Modulgruppe: {modulgruppe}",
        "",
        "PRIMARY ANALYSIS FOCUS:",
        "- Bauteil design and specifications",
        "- Bauteilkonzept (component concept)",
        "- Ausprägung (variant/version characteristics)",
        "",
        "SECONDARY DATA (may be outdated - use with caution):",
        "- Hauptbauteile, Schrauben, Muttern, Kontaktiert, Clipse",
        "- Sonstiges, Erklärung Sonstiges, Betriebsmittel",
        "- Beinhaltete Einzelbauteile, Beinhaltete Prozessschritte, Montageprozess",
        "",
        "MODEL DATA:",
        f"{m1} IEW eHPV: {model1_data.get('iew_ehpv', 0):.2f} min",
        f"{m2} IEW eHPV: {model2_data.get('iew_ehpv', 0):.2f} min",
        ""
    ]
    
    # Add available context data with priority on new fields
    bauteilkonzept_1 = model1_data.get('attributes', {}).get('Bauteilkonzept', '')
    bauteilkonzept_2 = model2_data.get('attributes', {}).get('Bauteilkonzept', '')
    auspraegung_1 = model1_data.get('attributes', {}).get('Ausprägung', '')
    auspraegung_2 = model2_data.get('attributes', {}).get('Ausprägung', '')
    
    if bauteilkonzept_1 or bauteilkonzept_2:
        prompt_parts.extend(["BAUTEILKONZEPT:", f"{m1}: {bauteilkonzept_1}", f"{m2}: {bauteilkonzept_2}", ""])
    
    if auspraegung_1 or auspraegung_2:
        prompt_parts.extend(["AUSPRÄGUNG:", f"{m1}: {auspraegung_1}", f"{m2}: {auspraegung_2}", ""])
    
    # Add secondary data with caution note
    assembly_1 = model1_data.get('attributes', {}).get('Montageprozess', '')
    assembly_2 = model2_data.get('attributes', {}).get('Montageprozess', '')
    
    if assembly_1 or assembly_2:
        prompt_parts.extend(["MONTAGEPROZESS (may be outdated):", f"{m1}: {assembly_1}", f"{m2}: {assembly_2}", ""])
    
    tools_1 = model1_data.get('attributes', {}).get('Betriebsmittel', '')
    tools_2 = model2_data.get('attributes', {}).get('Betriebsmittel', '')
    
    if tools_1 or tools_2:
        prompt_parts.extend(["BETRIEBSMITTEL (may be outdated):", f"{m1}: {tools_1}", f"{m2}: {tools_2}", ""])
    
    prompt_parts.extend([
        "ANALYSIS REQUIREMENTS:",
        "1. Start with a clear summary table showing key differences",
        "2. Focus analysis on Bauteilkonzept and Ausprägung as primary factors",
        "3. Note that assembly process data may be outdated",
        "4. Provide detailed reasoning based on available data",
        "5. Include manufacturing engineering insights",
        "",
        f"Respond in {language} with clear sections."
    ])
    
    prompt = "\n".join(prompt_parts)
    
    try:
        response = client.chat.completions.create(
            model=st.secrets["azure_openai"]["deployment_name"],
            messages=[
                {
                    "role": "system", 
                    "content": """You are a manufacturing engineering expert. Analyze EHPV differences with focus on 
                    Bauteilkonzept and Ausprägung. Always include a disclaimer about potentially outdated assembly process data. 
                    Provide structured analysis with summary table first, then detailed explanation."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=1500
        )
        
        # Add disclaimer to the response
        analysis_result = response.choices[0].message.content
        disclaimer = "\n\n---\n**Disclaimer**: Analysis of assembly processes, tooling, and fastener data is based on potentially outdated information. Primary focus should be on Bauteilkonzept and Ausprägung differences."
        
        return analysis_result + disclaimer
        
    except Exception as e:
        return f"Error generating AI analysis: {str(e)}"

# --- Data checks ---
if "ehpv_df" not in st.session_state:
    st.error("No dataset found - please upload your file in Step 1.")
    st.stop()
if "model1" not in st.session_state or "model2" not in st.session_state:
    st.error("Please select two Fahrzeug models in Step 2.")
    st.stop()

df = st.session_state["ehpv_df"].copy()
m1 = st.session_state["model1"]
m2 = st.session_state["model2"]

# --- Data processing ---
if "Berichtstyp/SA" in df.columns:
    df = df[df["Berichtstyp/SA"].astype(str).str.strip().str.upper() != "SA"]

required_cols = ["Fahrzeug", "Modulgruppe", "Bauteil", "Aktuelle IEW eHPV"]
missing = [col for col in required_cols if col not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

if "Bauteil (Englisch)" not in df.columns:
    df["Bauteil (Englisch)"] = df["Bauteil"]

df1 = df[df["Fahrzeug"] == m1].copy()
df2 = df[df["Fahrzeug"] == m2].copy()

df1["Aktuelle IEW eHPV"] = pd.to_numeric(df1["Aktuelle IEW eHPV"], errors="coerce").fillna(0.0)
df2["Aktuelle IEW eHPV"] = pd.to_numeric(df2["Aktuelle IEW eHPV"], errors="coerce").fillna(0.0)

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

delta_col = "Delta IEW eHPV"
merged[delta_col] = merged[f"Aktuelle IEW eHPV_{m2}"] - merged[f"Aktuelle IEW eHPV_{m1}"]

def derive_mg_family(value: object) -> str:
    if pd.isna(value):
        return "Other"
    match = re.match(r"(\d+)", str(value))
    if not match:
        return "Other"
    first_digit = match.group(1)[0]
    return f"MG{first_digit}x" if first_digit in "12345" else "Other"

merged["MG_Family"] = merged["Modulgruppe"].apply(derive_mg_family)

# --- Interactive filters ---
st.markdown("### Filters")
filtered = merged.copy()

all_families = sorted(filtered["MG_Family"].dropna().unique().tolist())
sel_families = st.multiselect("MG family:", all_families, default=all_families)
if sel_families:
    filtered = filtered[filtered["MG_Family"].isin(sel_families)]

available_modgroups = sorted(filtered["Modulgruppe"].dropna().unique().tolist())
sel_modgroups = st.multiselect("Modulgruppe:", available_modgroups)
if sel_modgroups:
    filtered = filtered[filtered["Modulgruppe"].isin(sel_modgroups)]

available_parts = sorted(filtered["Bauteil"].dropna().unique().tolist())
sel_parts = st.multiselect("Bauteil (German):", available_parts)
if sel_parts:
    filtered = filtered[filtered["Bauteil"].isin(sel_parts)]

# --- Detailed table with clickable selection ---
st.markdown("### Detailed IEW eHPV per Bauteil")
st.markdown("💡 **Click on any row to select it for AI analysis**")

detail = (
    filtered.groupby(
        ["MG_Family", "Modulgruppe", "Bauteil", "Bauteil (Englisch)"],
        as_index=False,
    )
    .agg(
        {
            f"Aktuelle IEW eHPV_{m1}": "sum",
            f"Aktuelle IEW eHPV_{m2}": "sum",
            delta_col: "sum",
        }
    )
)

def classify_scenario(row: pd.Series) -> str:
    has1 = row[f"Aktuelle IEW eHPV_{m1}"] > 0
    has2 = row[f"Aktuelle IEW eHPV_{m2}"] > 0
    if has1 and has2:
        return "Both models have this Bauteil"
    if has1:
        return f"{m1} only"
    if has2:
        return f"{m2} only"
    return "No IEW data"

detail["Scenario"] = detail.apply(classify_scenario, axis=1)
detail = detail.sort_values(["MG_Family", "Modulgruppe", delta_col], ascending=[True, True, False])

# Add row selection functionality
detail["_selected"] = False

# Initialize selected row in session state
if "selected_bauteil_detail" not in st.session_state:
    st.session_state.selected_bauteil_detail = None

# Display the table with selection
edited_df = st.data_editor(
    detail[
        ["MG_Family", "Modulgruppe", "Bauteil", "Bauteil (Englisch)", 
         f"Aktuelle IEW eHPV_{m1}", f"Aktuelle IEW eHPV_{m2}", 
         delta_col, "Scenario", "_selected"]
    ],
    column_config={
        "_selected": st.column_config.CheckboxColumn(
            "Select",
            help="Select row for AI analysis",
            default=False,
        ),
        "MG_Family": st.column_config.TextColumn("MG Family"),
        "Modulgruppe": st.column_config.TextColumn("Modulgruppe"),
        "Bauteil": st.column_config.TextColumn("Bauteil (German)"),
        "Bauteil (Englisch)": st.column_config.TextColumn("Bauteil (English)"),
        f"Aktuelle IEW eHPV_{m1}": st.column_config.NumberColumn(f"{m1} IEW", format="%.2f"),
        f"Aktuelle IEW eHPV_{m2}": st.column_config.NumberColumn(f"{m2} IEW", format="%.2f"),
        delta_col: st.column_config.NumberColumn("Delta", format="%+.2f"),
        "Scenario": st.column_config.TextColumn("Scenario"),
    },
    hide_index=True,
    use_container_width=True,
    height=400,
    disabled=["MG_Family", "Modulgruppe", "Bauteil", "Bauteil (Englisch)", 
              f"Aktuelle IEW eHPV_{m1}", f"Aktuelle IEW eHPV_{m2}", 
              delta_col, "Scenario"],
    key="detailed_table_selector"
)

# Find which row is selected and store it
selected_rows = edited_df[edited_df["_selected"]]
if not selected_rows.empty:
    selected_row = selected_rows.iloc[0]
    # Find the original row in detail dataframe
    original_row = detail[
        (detail["MG_Family"] == selected_row["MG_Family"]) &
        (detail["Modulgruppe"] == selected_row["Modulgruppe"]) &
        (detail["Bauteil"] == selected_row["Bauteil"])
    ].iloc[0]
    
    st.session_state.selected_bauteil_detail = {
        "key": f"{original_row['MG_Family']}|{original_row['Modulgruppe']}|{original_row['Bauteil']}",
        "mg_family": original_row["MG_Family"],
        "modulgruppe": original_row["Modulgruppe"],
        "bauteil": original_row["Bauteil"],
        "bauteil_en": original_row["Bauteil (Englisch)"],
        "iew_m1": original_row[f"Aktuelle IEW eHPV_{m1}"],
        "iew_m2": original_row[f"Aktuelle IEW eHPV_{m2}"],
        "delta": original_row[delta_col],
        "scenario": original_row["Scenario"],
    }
    st.success(f"Selected: {original_row['Bauteil']}")

# --- Detailed AI Analysis ---
st.markdown("---")
st.markdown("### Detailed AI Analysis")

# Context columns for AI
context_column_candidates = [
    "Bauteilkonzept", "Ausprägung",  # Primary fields
    "Hauptbauteile", "Schrauben", "Muttern", "Kontaktiert", "Clipse",
    "Sonstiges", "Erklärung Sonstiges", "Erklaerung Sonstiges",
    "Betriebsmittel", "Beinhaltete Einzelbauteile", "Beinhaltete Prozessschritte", "Montageprozess"
]
available_context_cols = [col for col in context_column_candidates if col in df.columns]

summary_state = st.session_state.setdefault("detailed_ai_summary", {"key": None, "content": None})

if detail.empty:
    st.info("No Bauteile available after applying filters.")
else:
    # Helper functions
    def safe_label(value: object, fallback: str) -> str:
        if pd.isna(value):
            return fallback
        text = str(value).strip()
        return text if text else fallback

    def to_float(value: object) -> float:
        try:
            return float(pd.to_numeric(value, errors="coerce"))
        except (TypeError, ValueError):
            return 0.0

    def collect_attribute_values(vehicle: str, selection: Dict[str, object]):
        mask = df["Fahrzeug"] == vehicle
        modul_value = selection.get("modulgruppe")
        part_value = selection.get("bauteil")
        if not pd.isna(modul_value):
            mask &= df["Modulgruppe"] == modul_value
        if not pd.isna(part_value):
            mask &= df["Bauteil"] == part_value
        subset = df.loc[mask, available_context_cols]
        textual = {}
        for col in available_context_cols:
            if col not in subset.columns:
                continue
            series = subset[col].dropna()
            if series.empty:
                continue
            as_strings = series.astype(str).str.strip()
            clean_values = [val for val in as_strings if val]
            if clean_values:
                textual[col] = clean_values[:10]
        return textual

    def flatten_attribute_values(values: Dict[str, List[str]]) -> Dict[str, str]:
        return {col: "; ".join(items) for col, items in values.items()}

    # Check if we have a selected row from the table
    selected_from_table = st.session_state.get("selected_bauteil_detail")
    
    # Build options for dropdown (fallback)
    options = [None]
    for _, row in detail.iterrows():
        options.append({
            "key": f"{row['MG_Family']}|{row['Modulgruppe']}|{row['Bauteil']}",
            "mg_family": row["MG_Family"],
            "modulgruppe": row["Modulgruppe"],
            "bauteil": row["Bauteil"],
            "bauteil_en": row["Bauteil (Englisch)"],
            "iew_m1": row[f"Aktuelle IEW eHPV_{m1}"],
            "iew_m2": row[f"Aktuelle IEW eHPV_{m2}"],
            "delta": row[delta_col],
            "scenario": row["Scenario"],
        })

    def format_option(option):
        if option is None:
            return "Select a Bauteil for detailed analysis"
        return f"{option['mg_family']} | {option['bauteil']}"

    # Auto-select from table selection or use dropdown
    default_index = 0
    if selected_from_table:
        # Find the matching option
        for i, option in enumerate(options):
            if option and option["key"] == selected_from_table["key"]:
                default_index = i
                break

    selected_option = st.selectbox(
        "Bauteil for AI analysis (or select from table above):",
        options,
        format_func=format_option,
        index=default_index
    )

    # Clear the table selection after using it
    if selected_from_table and "selected_bauteil_detail" in st.session_state:
        del st.session_state["selected_bauteil_detail"]

    if selected_option:
        # Display metrics
        value_m1 = to_float(selected_option["iew_m1"])
        value_m2 = to_float(selected_option["iew_m2"])
        delta_minutes = to_float(selected_option["delta"])

        cols = st.columns(3)
        cols[0].metric(f"{m1} IEW eHPV", f"{value_m1:.2f} min")
        cols[1].metric(f"{m2} IEW eHPV", f"{value_m2:.2f} min")
        cols[2].metric("Delta IEW eHPV", f"{delta_minutes:+.2f} min")

        # Language selection
        language_choice = st.radio("AI response language:", ["Deutsch", "English"], horizontal=True)

        # Collect context and generate analysis
        attrs_m1 = collect_attribute_values(m1, selected_option)
        attrs_m2 = collect_attribute_values(m2, selected_option)

        context_payload = {
            "part": selected_option["bauteil"],
            "modulgruppe": selected_option["modulgruppe"],
            "delta_minutes": delta_minutes,
            "model_1": {
                "name": m1,
                "iew_ehpv": value_m1,
                "attributes": flatten_attribute_values(attrs_m1),
            },
            "model_2": {
                "name": m2,
                "iew_ehpv": value_m2,
                "attributes": flatten_attribute_values(attrs_m2),
            },
        }

        current_key = (selected_option["key"], language_choice)
        if summary_state.get("key") != current_key:
            summary_state["key"] = current_key
            summary_state["content"] = None

        if st.button("Generate Detailed AI Analysis"):
            with st.spinner("Generating detailed analysis..."):
                summary_state["content"] = generate_ai_analysis(
                    m1=m1, m2=m2,
                    part=selected_option["bauteil"],
                    modulgruppe=selected_option["modulgruppe"],
                    delta_minutes=delta_minutes,
                    language=language_choice,
                    context=context_payload
                )

        if summary_state.get("content"):
            st.markdown("#### Detailed Analysis Result")
            st.markdown(summary_state["content"])

        # Show source data
        with st.expander("View Source Data"):
            part_rows = df[
                (df["Fahrzeug"].isin([m1, m2]))
                & (df["Modulgruppe"] == selected_option["modulgruppe"])
                & (df["Bauteil"] == selected_option["bauteil"])
            ]
            if not part_rows.empty:
                st.dataframe(part_rows, use_container_width=True)

# Navigation back
st.markdown("---")
if st.button("Back to Summary"):
    st.switch_page("pages/3_General_Summary.py")