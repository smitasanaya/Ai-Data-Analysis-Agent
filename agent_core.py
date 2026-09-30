"""
Phase 5 (updated): Agent Reasoning Loop — Gemini via the new google-genai SDK
Same ReAct pattern as before: user question -> Gemini decides which
tool(s) to call -> tool runs -> result fed back -> repeat until Gemini
gives a final answer.
"""

import os
from google import genai
from google.genai import types
import pandas as pd

from agent_tools import GEMINI_TOOL_DECLARATIONS, call_tool

# "gemini-flash-latest" is an auto-updated alias that always points to
# Google's current stable Flash model, so this won't break again when
# a specific dated model gets retired.
MODEL = "gemini-flash-latest"
MAX_TURNS = 6  # safety cap so the loop can't run forever


class DataAnalysisAgent:
    def __init__(self, df: pd.DataFrame, schema_info: dict, api_key: str = None):
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.df = df
        self.schema_info = schema_info
        self.tool = types.Tool(function_declarations=GEMINI_TOOL_DECLARATIONS)
        self.chart_paths = []

    def _system_prompt(self) -> str:
        cols = ", ".join(c["name"] for c in self.schema_info["columns"])
        return (
            "You are a data analysis agent. You have access to a dataset with "
            f"{self.schema_info['n_rows']} rows and columns: {cols}. "
            "Use the available tools to answer the user's question. "
            "Call tools as needed, then give a clear, concise natural-language "
            "answer that interprets the results — don't just dump raw numbers. "
            "If a chart was generated, mention what it shows."
        )

    def run(self, user_query: str, conversation_history: list = None) -> dict:
        """
        Run one query through the agent loop.
        Returns {"answer": str, "chart_paths": [...], "history": [...]}
        conversation_history is a google-genai chat history list, used to
        keep multi-turn context.
        """
        chat = self.client.chats.create(
            model=MODEL,
            config=types.GenerateContentConfig(
                tools=[self.tool],
                system_instruction=self._system_prompt(),
            ),
            history=conversation_history or [],
        )
        self.chart_paths = []

        response = chat.send_message(user_query)

        for _ in range(MAX_TURNS):
            parts = response.candidates[0].content.parts
            function_calls = [p.function_call for p in parts if p.function_call]

            if not function_calls:
                final_text = response.text
                return {"answer": final_text, "chart_paths": self.chart_paths, "history": chat.get_history()}

            # Execute every tool call Gemini requested in this turn
            function_response_parts = []
            for fc in function_calls:
                args = dict(fc.args)
                result = call_tool(fc.name, args, self.df)
                if isinstance(result, dict) and "chart_path" in result:
                    self.chart_paths.append(result["chart_path"])
                function_response_parts.append(
                    types.Part.from_function_response(name=fc.name, response={"result": result})
                )

            response = chat.send_message(function_response_parts)

        return {
            "answer": "I wasn't able to reach a final answer within the step limit — try a more specific question.",
            "chart_paths": self.chart_paths,
            "history": chat.get_history(),
        }
