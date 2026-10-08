from state import AgentState


state = AgentState(
    "Tell me the status of order ABC"
)

print("User Query:", state.user_query)
print("Current Step:", state.current_step)
print("Tool Calls:", state.tool_calls)
print("Tool Results:", state.tool_results)
print("Retry Count:", state.retry_count)
print("Final Answer:", state.final_answer)