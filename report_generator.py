"""
Phase 6 (updated): Report Generation — Gemini via the new google-genai SDK
Runs a fixed exploratory analysis pass over the whole dataset and asks
Gemini to synthesize the findings into a written summary.
"""

import os
from google import genai
import pandas as pd

from agent_tools import (
    get_summary_stats, detect_outliers, plot_correlation_heatmap,
    plot_distribution, get_data_quality_score
)

MODEL = "gemini-flash-latest"


def run_auto_eda(df: pd.DataFrame, schema_info: dict) -> dict:
    """
    Runs summary stats + outlier checks on every numeric column, a data
    quality score, and generates a correlation heatmap. Returns raw
    findings + chart paths.
    """
    findings = {
        "summary_stats": get_summary_stats(df),
        "quality": get_data_quality_score(df),
        "outliers": {},
        "charts": [],
    }

    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        findings["outliers"][col] = detect_outliers(df, col)

    heatmap = plot_correlation_heatmap(df)
    if "chart_path" in heatmap:
        findings["charts"].append(heatmap["chart_path"])

    # Distribution plots for up to the first 4 numeric columns (keeps report short)
    for col in list(numeric_cols)[:4]:
        dist = plot_distribution(df, col)
        if "chart_path" in dist:
            findings["charts"].append(dist["chart_path"])

    return findings


def generate_written_summary(schema_info: dict, findings: dict, api_key: str = None) -> str:
    """
    Sends the raw findings to Gemini and asks for a written, human-readable
    EDA summary — the kind you'd put at the top of a report.
    """
    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

    prompt = (
        "Here is the schema and exploratory analysis findings for a dataset. "
        "Write a clear, well-organized summary (use short sections) covering: "
        "data quality issues, notable outliers, and any interesting patterns. "
        "Keep it factual and grounded only in the numbers given.\n\n"
        f"SCHEMA:\n{schema_info}\n\nFINDINGS:\n{findings}"
    )

    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text
