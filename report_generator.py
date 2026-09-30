"""
Phase 6 (updated): Report Generation — Gemini via the new google-genai SDK
Runs a fixed exploratory analysis pass over the whole dataset and asks
Gemini to synthesize the findings into a written summary.
Includes retry + fallback-model handling for transient server errors.
"""

import os
import time
from google import genai
from google.genai import errors as genai_errors
import pandas as pd

from agent_tools import (
    get_summary_stats, detect_outliers, plot_correlation_heatmap,
    plot_distribution, get_data_quality_score
)

# Primary model, and a fallback to try if the primary is overloaded/unavailable.
PRIMARY_MODEL = "gemini-flash-latest"
FALLBACK_MODEL = "gemini-2.5-flash"
MAX_RETRIES = 3


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
    EDA summary. Retries on transient server errors, then falls back to a
    secondary model if the primary keeps failing.
    """
    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

    prompt = (
        "Here is the schema and exploratory analysis findings for a dataset. "
        "Write a clear, well-organized summary (use short sections) covering: "
        "data quality issues, notable outliers, and any interesting patterns. "
        "Keep it factual and grounded only in the numbers given.\n\n"
        f"SCHEMA:\n{schema_info}\n\nFINDINGS:\n{findings}"
    )

    last_error = None
    for model in (PRIMARY_MODEL, FALLBACK_MODEL):
        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text
            except genai_errors.ServerError as e:
                last_error = e
                time.sleep(2 * (attempt + 1))  # brief backoff before retrying
            except genai_errors.ClientError as e:
                # 4xx errors (bad key, bad request) won't fix themselves on retry
                raise RuntimeError(f"Gemini request failed: {e}") from e

    raise RuntimeError(
        "Gemini's servers are currently unavailable after multiple retries. "
        "This is usually temporary — please try again in a minute. "
        f"(Last error: {last_error})"
    )
