# AI Agent for Automated Data Analysis

An LLM-powered agent (now running on **Google Gemini**, free tier) that takes a CSV/Excel file and either:
1. Generates a one-click automated EDA report, or
2. Answers free-form natural-language questions about the data by calling analysis tools (summary stats, correlation, outlier detection, trend forecasting, category breakdowns, plots).

## What's new in this version
- **Gemini backend** instead of Claude — free API key, no billing required for a student project
- **Two new analysis tools**: `forecast_trend` (simple linear trend projection) and `get_top_categories` (categorical breakdowns)
- **Data quality score**: an instant 0–100 score shown as a metric card after upload
- **Search history**: every question you ask is saved locally (`search_history.json`) and shown in the sidebar — click "Ask again" to replay one
- **Suggested questions**: clickable chips generated automatically from your dataset's actual columns
- **Redesigned UI**: gradient header, styled metric cards, custom theme (`.streamlit/config.toml`), download buttons for the report and chat history

## Project Structure

| File | Phase | Purpose |
|---|---|---|
| `data_ingestion.py` | Phase 3 | Load, profile, and auto-clean uploaded datasets |
| `agent_tools.py` | Phase 4 | Analysis functions the agent can call (now includes forecasting, category breakdowns, quality score) + Gemini tool schemas |
| `agent_core.py` | Phase 5 | The ReAct reasoning loop, using Gemini's function-calling API |
| `report_generator.py` | Phase 6 | Automated one-click EDA report generation (Gemini) |
| `history_manager.py` | Phase D | Persists search history to a local JSON file, keyed per dataset |
| `suggestions.py` | Phase E | Generates auto-suggested example questions from the dataset schema |
| `app.py` | Phase 7 | Streamlit UI tying everything together, with the new design |
| `.streamlit/config.toml` | Phase F | Custom color theme |

## Setup

```bash
pip install -r requirements.txt
```

Get a **free** Gemini API key at [aistudio.google.com](https://aistudio.google.com) → "Get API key" → "Create API key". No credit card needed for the free tier.

You can either:
- Set it as an environment variable: `export GEMINI_API_KEY=your_key_here` (Mac/Linux) or `set GEMINI_API_KEY=your_key_here` (Windows)
- Or paste it into the sidebar text box when the app is running

## Run

```bash
streamlit run app.py
```

Then open the local URL it prints, upload a CSV/Excel file, and either generate an auto-report or start chatting with the agent about your data.

## How the agent loop works (Phase 5, for your report/viva)

1. User asks a question ("what's the trend in sales over time?")
2. The question + tool schemas + dataset schema are sent to Gemini
3. Gemini decides which tool(s) to call and with what arguments (the "reasoning" step)
4. The app executes the tool in Python and sends the result back to Gemini
5. Gemini either calls another tool (if it needs more info) or writes a final natural-language answer
6. This loops up to `MAX_TURNS` times as a safety cap

This is the standard **ReAct (Reason + Act)** pattern used in most LLM agent frameworks.

## For a 5-person team

Share one Gemini API key privately (never commit it to GitHub or post it publicly) — each teammate pastes it into their own local run of the app. `search_history.json` is local to whoever runs the app; it isn't shared automatically between teammates unless you share that file too.

## Phase 8: Testing (suggested, not automated here)

Test with datasets of varying messiness:
- A clean dataset (sanity check)
- One with missing values / duplicates (tests auto-cleaning)
- One with mixed types in a column (tests type coercion)
- A column with a clear trend (tests `forecast_trend`)
- Ambiguous questions ("tell me something interesting") to see how the agent handles open-ended prompts

## Phase 9: Further extension ideas
- Persist history centrally (e.g. SQLite) so all 5 teammates share one history
- Add user labels to history entries (who asked what)
- Export the Auto EDA Report as a formatted PDF/Word doc instead of plain text
