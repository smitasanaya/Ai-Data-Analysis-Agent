"""
Phase 3: Data Ingestion Module
Loads raw CSV/Excel files, profiles the schema, and applies basic
auto-cleaning. This is the first thing that runs before the agent
ever sees the data.
"""

import pandas as pd
import numpy as np


def load_data(file_path_or_buffer, file_type: str = None) -> pd.DataFrame:
    """
    Load a CSV or Excel file into a DataFrame.

    file_path_or_buffer: path string OR a Streamlit UploadedFile object
    file_type: 'csv' or 'excel'. If None, inferred from filename.
    """
    if file_type is None:
        name = getattr(file_path_or_buffer, "name", str(file_path_or_buffer))
        file_type = "excel" if name.endswith((".xlsx", ".xls")) else "csv"

    if file_type == "csv":
        df = pd.read_csv(file_path_or_buffer)
    else:
        df = pd.read_excel(file_path_or_buffer)

    return df


def get_schema_info(df: pd.DataFrame) -> dict:
    """
    Profile the dataset: column types, missing values, uniqueness.
    This profile is what gets shown to the LLM agent so it understands
    what it's working with before answering questions.
    """
    info = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": []
    }

    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "missing_pct": round(float(df[col].isna().mean() * 100), 2),
            "n_unique": int(df[col].nunique()),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["min"] = float(df[col].min()) if df[col].notna().any() else None
            col_info["max"] = float(df[col].max()) if df[col].notna().any() else None
            col_info["mean"] = float(df[col].mean()) if df[col].notna().any() else None
        info["columns"].append(col_info)

    info["duplicate_rows"] = int(df.duplicated().sum())
    return info


def auto_clean(df: pd.DataFrame, drop_duplicates: bool = True,
                numeric_fill: str = "median") -> pd.DataFrame:
    """
    Basic automatic cleaning pass:
    - Strips whitespace from string columns
    - Drops exact duplicate rows
    - Fills missing numeric values (median/mean) — flags rest for the agent
    - Attempts to convert object columns that are actually numeric/dates
    """
    df = df.copy()

    # Strip whitespace on string/object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # Try to auto-convert object columns that are secretly numeric
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() / max(len(df), 1) > 0.9:
            df[col] = converted

    if drop_duplicates:
        df = df.drop_duplicates()

    # Fill numeric missing values
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isna().any():
            fill_value = df[col].median() if numeric_fill == "median" else df[col].mean()
            df[col] = df[col].fillna(fill_value)

    return df


def summarize_cleaning(before: pd.DataFrame, after: pd.DataFrame) -> str:
    """Human-readable summary of what auto_clean changed."""
    rows_removed = len(before) - len(after)
    lines = [
        f"Rows before cleaning: {len(before)}",
        f"Rows after cleaning: {len(after)} ({rows_removed} duplicates removed)",
        f"Missing values before: {int(before.isna().sum().sum())}",
        f"Missing values after: {int(after.isna().sum().sum())}",
    ]
    return "\n".join(lines)
