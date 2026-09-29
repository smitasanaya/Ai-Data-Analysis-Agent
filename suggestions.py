"""
Phase E (new): Smart Suggestions
Generates relevant example questions based on the dataset's actual
columns, so users aren't staring at a blank chat box. Purely heuristic
(no LLM call) — instant and free to run.
"""

import pandas as pd


def suggest_questions(df: pd.DataFrame, max_suggestions: int = 6) -> list:
    """Build a short list of relevant example questions for this dataset."""
    suggestions = []
    numeric_cols = list(df.select_dtypes(include="number").columns)
    categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)

    suggestions.append("How clean is this dataset overall?")

    if len(numeric_cols) >= 2:
        suggestions.append(f"What's the correlation between {numeric_cols[0]} and {numeric_cols[1]}?")

    if numeric_cols:
        suggestions.append(f"Are there any outliers in {numeric_cols[0]}?")
        suggestions.append(f"Show me the distribution of {numeric_cols[0]}")
        if len(df) >= 5:
            suggestions.append(f"What's the trend in {numeric_cols[0]}?")

    if categorical_cols:
        suggestions.append(f"What are the top categories in {categorical_cols[0]}?")

    suggestions.append("Give me a summary of all the numeric columns")

    return suggestions[:max_suggestions]
