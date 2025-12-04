import pandas as pd

def read_excel_safely(file):
    try:
        df = pd.read_excel(file, engine="openpyxl")
        return df, None
    except Exception as e:
        return None, f"Couldn't read Excel file: {e}"
