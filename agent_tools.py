"""
Phase 4 (updated): Agent Toolset
Tool implementations + Gemini-format function declarations.
Charts are saved to disk and their file paths returned so the UI
layer (Streamlit) can display them.
"""

import os
import uuid
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CHART_DIR = "generated_charts"
os.makedirs(CHART_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------

def get_summary_stats(df: pd.DataFrame, columns: list = None) -> dict:
    """Return descriptive statistics for numeric columns."""
    subset = df[columns] if columns else df.select_dtypes(include="number")
    if subset.empty:
        return {"error": "No numeric columns found for the given selection."}
    return subset.describe().round(3).to_dict()


def check_correlation(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """Compute Pearson correlation between two numeric columns."""
    for c in (col1, col2):
        if c not in df.columns:
            return {"error": f"Column '{c}' not found."}
    corr = df[col1].corr(df[col2])
    return {"col1": col1, "col2": col2, "pearson_correlation": round(float(corr), 4)}


def detect_outliers(df: pd.DataFrame, column: str) -> dict:
    """Detect outliers in a numeric column using the IQR method."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return {
        "column": column,
        "lower_bound": round(float(lower), 3),
        "upper_bound": round(float(upper), 3),
        "n_outliers": int(len(outliers)),
        "outlier_pct": round(len(outliers) / max(len(df), 1) * 100, 2),
    }


def filter_data(df: pd.DataFrame, column: str, operator: str, value) -> dict:
    """Filter rows by a simple condition. operator: >, <, >=, <=, ==, !="""
    ops = {
        ">": lambda s, v: s > v, "<": lambda s, v: s < v,
        ">=": lambda s, v: s >= v, "<=": lambda s, v: s <= v,
        "==": lambda s, v: s == v, "!=": lambda s, v: s != v,
    }
    if column not in df.columns or operator not in ops:
        return {"error": "Invalid column or operator."}
    try:
        value = float(value)
    except (TypeError, ValueError):
        pass
    filtered = df[ops[operator](df[column], value)]
    return {"matching_rows": int(len(filtered)), "preview": filtered.head(5).to_dict(orient="records")}


def get_top_categories(df: pd.DataFrame, column: str, top_n: int = 10) -> dict:
    """Get the most frequent values in a column (categorical breakdown)."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    counts = df[column].value_counts().head(top_n)
    return {
        "column": column,
        "top_values": counts.to_dict(),
        "n_unique_total": int(df[column].nunique()),
    }


def forecast_trend(df: pd.DataFrame, column: str, periods: int = 5) -> dict:
    """
    Simple linear-trend forecast for a numeric column, projecting the next
    N points based on row order (treats row index as the time axis).
    Uses ordinary least squares (degree-1 polynomial fit) — a lightweight,
    explainable forecasting approach suitable for a quick trend read.
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' not found or not numeric."}
    series = df[column].dropna().reset_index(drop=True)
    if len(series) < 5:
        return {"error": "Not enough data points to forecast (need at least 5)."}

    x = np.arange(len(series))
    slope, intercept = np.polyfit(x, series.values, 1)
    future_x = np.arange(len(series), len(series) + periods)
    forecast = (slope * future_x + intercept).round(3).tolist()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, series.values, label="Historical", color="#3B5F8A")
    ax.plot(future_x, forecast, label="Forecast", color="#B04A4A", linestyle="--", marker="o")
    ax.set_title(f"Trend Forecast: {column}")
    ax.legend()
    path = os.path.join(CHART_DIR, f"forecast_{column}_{uuid.uuid4().hex[:6]}.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    trend_direction = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "flat")
    return {
        "column": column,
        "trend_direction": trend_direction,
        "slope_per_row": round(float(slope), 4),
        "forecast_next_values": forecast,
        "chart_path": path,
    }


def get_data_quality_score(df: pd.DataFrame) -> dict:
    """
    Compute a simple composite data-quality score (0-100) based on
    missing-value rate and duplicate-row rate. Higher is better.
    """
    missing_pct = float(df.isna().mean().mean() * 100)
    duplicate_pct = float(df.duplicated().mean() * 100)
    score = max(0.0, 100 - (missing_pct * 0.6) - (duplicate_pct * 0.4))
    return {
        "quality_score": round(score, 1),
        "missing_pct": round(missing_pct, 2),
        "duplicate_pct": round(duplicate_pct, 2),
    }


def plot_distribution(df: pd.DataFrame, column: str) -> dict:
    """Plot a histogram for a numeric column, or a bar count for categorical."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    fig, ax = plt.subplots(figsize=(6, 4))
    if pd.api.types.is_numeric_dtype(df[column]):
        sns.histplot(df[column].dropna(), kde=True, ax=ax)
    else:
        df[column].value_counts().head(15).plot(kind="bar", ax=ax)
    ax.set_title(f"Distribution of {column}")
    path = os.path.join(CHART_DIR, f"{column}_{uuid.uuid4().hex[:6]}.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"chart_path": path}


def plot_correlation_heatmap(df: pd.DataFrame) -> dict:
    """Plot a correlation heatmap across all numeric columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return {"error": "Need at least 2 numeric columns for a heatmap."}
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    path = os.path.join(CHART_DIR, f"heatmap_{uuid.uuid4().hex[:6]}.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"chart_path": path}


# ---------------------------------------------------------------------
# Tool schemas — Gemini function-calling format
# ---------------------------------------------------------------------

GEMINI_TOOL_DECLARATIONS = [
    {
        "name": "get_summary_stats",
        "description": "Get descriptive statistics (mean, std, min, max, quartiles) for numeric columns. Use for general 'summarize the data' questions.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "columns": {"type": "ARRAY", "items": {"type": "STRING"},
                            "description": "Optional list of specific columns. Omit for all numeric columns."}
            }
        }
    },
    {
        "name": "check_correlation",
        "description": "Compute the Pearson correlation coefficient between two numeric columns.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "col1": {"type": "STRING"},
                "col2": {"type": "STRING"}
            },
            "required": ["col1", "col2"]
        }
    },
    {
        "name": "detect_outliers",
        "description": "Detect outliers in a numeric column using the IQR method.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"column": {"type": "STRING"}},
            "required": ["column"]
        }
    },
    {
        "name": "filter_data",
        "description": "Filter rows of the dataset where a column meets a condition (e.g. age > 30).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "column": {"type": "STRING"},
                "operator": {"type": "STRING", "enum": [">", "<", ">=", "<=", "==", "!="]},
                "value": {"type": "STRING", "description": "Value to compare against."}
            },
            "required": ["column", "operator", "value"]
        }
    },
    {
        "name": "get_top_categories",
        "description": "Get the most frequent values in a column — useful for categorical breakdowns (e.g. 'what are the top product categories?').",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "column": {"type": "STRING"},
                "top_n": {"type": "INTEGER", "description": "How many top values to return. Defaults to 10."}
            },
            "required": ["column"]
        }
    },
    {
        "name": "forecast_trend",
        "description": "Project a simple linear trend forecast for a numeric column into the next N rows. Use for questions like 'what will this look like going forward' or 'is this trending up or down'.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "column": {"type": "STRING"},
                "periods": {"type": "INTEGER", "description": "How many future points to forecast. Defaults to 5."}
            },
            "required": ["column"]
        }
    },
    {
        "name": "get_data_quality_score",
        "description": "Compute an overall data quality score (0-100) based on missing values and duplicate rows. Use for questions like 'how clean is my data' or 'how good is this dataset'.",
        "parameters": {"type": "OBJECT", "properties": {}}
    },
    {
        "name": "plot_distribution",
        "description": "Generate a distribution plot (histogram for numeric, bar chart for categorical) of a single column.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"column": {"type": "STRING"}},
            "required": ["column"]
        }
    },
    {
        "name": "plot_correlation_heatmap",
        "description": "Generate a correlation heatmap across all numeric columns in the dataset.",
        "parameters": {"type": "OBJECT", "properties": {}}
    },
]

TOOL_FUNCTIONS = {
    "get_summary_stats": get_summary_stats,
    "check_correlation": check_correlation,
    "detect_outliers": detect_outliers,
    "filter_data": filter_data,
    "get_top_categories": get_top_categories,
    "forecast_trend": forecast_trend,
    "get_data_quality_score": get_data_quality_score,
    "plot_distribution": plot_distribution,
    "plot_correlation_heatmap": plot_correlation_heatmap,
}


def call_tool(name: str, tool_input: dict, df: pd.DataFrame):
    """Dispatch a tool call by name to its implementation."""
    if name not in TOOL_FUNCTIONS:
        return {"error": f"Unknown tool '{name}'"}
    return TOOL_FUNCTIONS[name](df, **tool_input)
