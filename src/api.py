import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent_langgraph import app as agent_app


api = FastAPI(
    title="Olist Agentic AI API",
    description="FastAPI wrapper around the LangGraph Olist agent",
    version="1.0.0"
)


class AskRequest(BaseModel):
    query: str


@api.get("/health")
def health():
    return {
        "status": "ok",
        "service": "olist-agentic-ai"
    }


@api.post("/ask")
def ask(request: AskRequest):

    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    request_start = time.perf_counter()

    system_message = {
        "role": "system",
        "content": """
You are an Olist e-commerce customer support agent.

Use tools when factual information is required.

Rules:

1. Order information → use order tools.
2. Customer order history → use customer tools.
3. Delivery delay → use delivery tool.
4. Customer repeat-order prediction → use
   predict_customer_repeat_order.
5. Company policies → use search_knowledge.
6. Never invent order/customer information.
7. Give a concise final answer.
"""
    }

    initial_state = {
        "user_query": request.query,
        "messages": [
            system_message,
            {
                "role": "user",
                "content": request.query
            }
        ],
        "tool_result": None,
        "final_answer": "",
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
        "llm_calls": 0,
        "total_llm_latency": 0.0,
        "total_tool_latency": 0.0,
        "errors": []
    }

    try:
        result = agent_app.invoke(initial_state)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {str(e)}"
        )

    total_request_latency = (
        time.perf_counter() - request_start
    )

    messages = result.get("messages", [])

    if not messages:
        raise HTTPException(
            status_code=500,
            detail="Agent returned no messages."
        )

    final_message = messages[-1]

    errors = result.get("errors", [])

    return {
        "success": len(errors) == 0,
        "query": request.query,
        "answer": final_message.content,

        "metrics": {
            "llm_calls": result.get("llm_calls", 0),
            "input_tokens": result.get(
                "total_input_tokens", 0
            ),
            "output_tokens": result.get(
                "total_output_tokens", 0
            ),
            "total_tokens": result.get(
                "total_tokens", 0
            ),
            "estimated_cost_usd": round(
                result.get("total_cost", 0.0),
                8
            ),
            "total_llm_latency_seconds": round(
                result.get("total_llm_latency", 0.0),
                4
            ),
            "total_tool_latency_seconds": round(
                result.get("total_tool_latency", 0.0),
                4
            ),
            "total_latency_seconds": round(
                total_request_latency,
                4
            )
        },

        "errors": errors
    }