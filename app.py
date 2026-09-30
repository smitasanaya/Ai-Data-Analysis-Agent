"""
Phase 7 (updated): Streamlit UI
Wires together ingestion -> Gemini agent -> report generation, with:
- A redesigned, styled interface
- Data quality metric cards
- Auto-suggested question chips
- Persistent search history (per dataset)
Run with: streamlit run app.py
"""

import streamlit as st

from data_ingestion import load_data, get_schema_info, auto_clean, summarize_cleaning
from agent_core import DataAnalysisAgent
from report_generator import run_auto_eda, generate_written_summary
from agent_tools import get_data_quality_score
from suggestions import suggest_questions
import history_manager as history

st.set_page_config(page_title="AI Data Analysis Agent", page_icon="🤖", layout="wide")

# ---------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #6C63FF 0%, #3B5F8A 100%);
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
}
.hero h1 { color: white; margin: 0; font-size: 30px; }
.hero p { color: #E8E6FF; margin: 6px 0 0 0; font-size: 15px; }

div[data-testid="stMetric"] {
    background-color: #1A1D29;
    border: 1px solid #2E3140;
    border-radius: 12px;
    padding: 14px 16px;
}

.suggestion-chip {
    display: inline-block;
    background-color: #1A1D29;
    border: 1px solid #6C63FF55;
    border-radius: 20px;
    padding: 6px 14px;
    margin: 4px 6px 4px 0;
    font-size: 13px;
    color: #C9C6FF;
}

.history-item {
    background-color: #1A1D29;
    border-left: 3px solid #6C63FF;
    border-radius: 6px;
    padding: 8px 10px;
    margin-bottom: 8px;
    font-size: 12.5px;
}
.history-item .q { color: #F5F5F7; font-weight: 600; }
.history-item .t { color: #888; font-size: 11px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🤖 AI Agent for Automated Data Analysis</h1>
  <p>Upload a dataset, get an instant EDA report, or ask questions in plain English.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.schema_info = None
    st.session_state.chat_history = []       # for display
    st.session_state.agent_messages = []      # Gemini-format history
    st.session_state.prefill_query = None

# ---------------------------------------------------------------------
# Sidebar: setup + history
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = st.text_input("Gemini API Key", type="password", help="Get one free at aistudio.google.com")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None and st.session_state.df is None:
        raw_df = load_data(uploaded_file)
        cleaned_df = auto_clean(raw_df)
        st.session_state.df = cleaned_df
        st.session_state.schema_info = get_schema_info(cleaned_df)
        st.success("File loaded and auto-cleaned.")
        with st.expander("Cleaning summary"):
            st.text(summarize_cleaning(raw_df, cleaned_df))

    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.df = None
        st.session_state.chat_history = []
        st.session_state.agent_messages = []
        st.rerun()

    if st.session_state.df is not None:
        st.divider()
        st.header("🕘 Search History")
        past = history.get_history(st.session_state.schema_info)
        if not past:
            st.caption("No questions asked yet.")
        else:
            for i, entry in enumerate(past[:15]):
                st.markdown(
                    f"""<div class="history-item">
                        <div class="q">{entry['question']}</div>
                        <div class="t">{entry['timestamp']}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if st.button("↺ Ask again", key=f"replay_{i}", use_container_width=True):
                    st.session_state.prefill_query = entry["question"]
            if st.button("🗑️ Clear history", use_container_width=True):
                history.clear_history(st.session_state.schema_info)
                st.rerun()

# ---------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------
if st.session_state.df is None:
    st.info("👈 Upload a dataset from the sidebar to get started.")
    st.stop()

df = st.session_state.df
schema_info = st.session_state.schema_info

# --- Data quality metric cards ---
quality = get_data_quality_score(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{schema_info['n_rows']:,}")
c2.metric("Columns", schema_info["n_cols"])
c3.metric("Quality Score", f"{quality['quality_score']}/100")
c4.metric("Missing Data", f"{quality['missing_pct']}%")

st.subheader("📋 Data Preview")
st.dataframe(df.head(20), use_container_width=True)

tab_report, tab_chat = st.tabs(["📊 Auto EDA Report", "💬 Ask the Agent"])

# --- Tab 1: one-click automated report ---
with tab_report:
    if st.button("✨ Generate Report", type="primary"):
        if not api_key:
            st.error("Enter your Gemini API key in the sidebar first.")
        else:
            try:
                with st.spinner("Running exploratory analysis..."):
                    findings = run_auto_eda(df, schema_info)
                    summary = generate_written_summary(schema_info, findings, api_key)
                st.markdown(summary)
                cols = st.columns(2)
                for i, chart_path in enumerate(findings["charts"]):
                    cols[i % 2].image(chart_path)
                st.download_button("⬇️ Download report as text", summary, file_name="eda_report.txt")
            except RuntimeError as e:
                st.error(f"⚠️ Couldn't generate the report: {e}")

# --- Tab 2: interactive chat with the agent ---
with tab_chat:
    st.caption("Try one of these:")
    for q in suggest_questions(df):
        if st.button(q, key=f"sugg_{q}"):
            st.session_state.prefill_query = q

    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(text)

    query_to_run = st.session_state.prefill_query or st.chat_input("Ask a question about your data...")
    st.session_state.prefill_query = None

    if query_to_run:
        if not api_key:
            st.error("Enter your Gemini API key in the sidebar first.")
        else:
            st.session_state.chat_history.append(("user", query_to_run))
            with st.chat_message("user"):
                st.markdown(query_to_run)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Thinking..."):
                        agent = DataAnalysisAgent(df, schema_info, api_key)
                        result = agent.run(query_to_run, st.session_state.agent_messages)
                    st.markdown(result["answer"])
                    for chart_path in result["chart_paths"]:
                        st.image(chart_path)
                    st.session_state.chat_history.append(("assistant", result["answer"]))
                    st.session_state.agent_messages = result["history"]
                    history.add_entry(schema_info, query_to_run, result["answer"])
                except RuntimeError as e:
                    st.error(f"⚠️ Couldn't get a response: {e}")

    if st.session_state.chat_history:
        chat_text = "\n\n".join(f"{role.upper()}: {text}" for role, text in st.session_state.chat_history)
        st.download_button("⬇️ Download chat as text", chat_text, file_name="chat_history.txt")
