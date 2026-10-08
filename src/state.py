from typing import Any


class AgentState:
    def __init__(self, user_query: str):

        self.user_query = user_query

        # LLM conversation history
        self.messages = []

        # Agent execution tracking
        self.current_step = 0

        # Tool tracking
        self.tool_calls = []
        self.tool_results = []

        # Error and retry tracking
        self.errors = []
        self.retry_count = 0

        # Final response
        self.final_answer = None

    def add_tool_call(self, tool_name: str, arguments: dict):
        self.tool_calls.append({
            "tool_name": tool_name,
            "arguments": arguments
        })

    def add_tool_result(self, tool_name: str, result: Any):
        self.tool_results.append({
            "tool_name": tool_name,
            "result": result
        })

    def add_error(self, error: str):
        self.errors.append(error)

    def increment_step(self):
        self.current_step += 1