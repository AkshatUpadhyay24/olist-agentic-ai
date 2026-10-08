
import os
import json
import time
from typing import TypedDict, Any

from dotenv import load_dotenv
from openai import OpenAI

from langgraph.graph import StateGraph, START, END

from tools.order_tools import get_order_details
from tools.customer_tools import get_customer_orders
from tools.delivery_tools import calculate_delivery_delay
from tools.customer_prediction_tool import predict_customer_repeat_order
from rag.rag_tool import search_knowledge


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL_NAME = "gpt-4o-mini"


# ============================================================
# OPENAI PRICING
# ============================================================

INPUT_COST_PER_1M = 0.15
OUTPUT_COST_PER_1M = 0.60


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict):
    user_query: str
    messages: list
    tool_result: Any
    final_answer: str

    # Token tracking
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int

    # Cost tracking
    total_cost: float

    # LLM tracking
    llm_calls: int

    # Latency tracking
    total_llm_latency: float
    total_tool_latency: float

    # Error tracking
    errors: list


# ============================================================
# TOOL DEFINITIONS
# ============================================================

tools = [

    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": "Get details of an Olist order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string"
                    }
                },
                "required": ["order_id"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_customer_orders",
            "description": "Get all orders placed by an Olist customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string"
                    }
                },
                "required": ["customer_id"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "calculate_delivery_delay",
            "description": "Check whether an order was delivered late.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string"
                    }
                },
                "required": ["order_id"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "predict_customer_repeat_order",
            "description": (
                "Predict whether a customer is likely to "
                "place another order based on historical orders."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string"
                    }
                },
                "required": ["customer_id"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Search company policies and knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# ============================================================
# TOOL MAP
# ============================================================

available_tools = {
    "get_order_details": get_order_details,
    "get_customer_orders": get_customer_orders,
    "calculate_delivery_delay": calculate_delivery_delay,
    "predict_customer_repeat_order": predict_customer_repeat_order,
    "search_knowledge": search_knowledge
}


# ============================================================
# AGENT NODE
# ============================================================

def agent_node(state: AgentState):

    print("\n========================================")
    print("LANGGRAPH → AGENT NODE")
    print("========================================")

    messages = state["messages"]

    # Start LLM timer
    llm_start = time.perf_counter()

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

    except Exception as e:

        print("\nLLM ERROR:", str(e))

        errors = state.get("errors", [])
        errors.append(str(e))

        return {
            "errors": errors
        }

    # End LLM timer
    llm_latency = time.perf_counter() - llm_start

    # ========================================================
    # TOKEN + COST
    # ========================================================

    usage = response.usage

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    call_cost = 0.0

    if usage:

        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        input_cost = (
            input_tokens / 1_000_000
        ) * INPUT_COST_PER_1M

        output_cost = (
            output_tokens / 1_000_000
        ) * OUTPUT_COST_PER_1M

        call_cost = input_cost + output_cost

        print("\nTOKEN + COST USAGE")
        print("--------------------------")
        print("Input tokens  :", input_tokens)
        print("Output tokens :", output_tokens)
        print("Total tokens  :", total_tokens)
        print("Input cost    : $", round(input_cost, 8))
        print("Output cost   : $", round(output_cost, 8))
        print("Call cost     : $", round(call_cost, 8))

    print("LLM latency   :", round(llm_latency, 4), "seconds")

    # ========================================================
    # UPDATE TOTALS
    # ========================================================

    new_total_input_tokens = (
        state.get("total_input_tokens", 0)
        + input_tokens
    )

    new_total_output_tokens = (
        state.get("total_output_tokens", 0)
        + output_tokens
    )

    new_total_tokens = (
        state.get("total_tokens", 0)
        + total_tokens
    )

    new_total_cost = (
        state.get("total_cost", 0.0)
        + call_cost
    )

    new_llm_calls = (
        state.get("llm_calls", 0)
        + 1
    )

    new_total_llm_latency = (
        state.get("total_llm_latency", 0.0)
        + llm_latency
    )

    # ========================================================
    # ADD ASSISTANT MESSAGE
    # ========================================================

    message = response.choices[0].message

    messages.append(message)

    return {
        "messages": messages,

        "total_input_tokens":
            new_total_input_tokens,

        "total_output_tokens":
            new_total_output_tokens,

        "total_tokens":
            new_total_tokens,

        "total_cost":
            new_total_cost,

        "llm_calls":
            new_llm_calls,

        "total_llm_latency":
            new_total_llm_latency
    }


# ============================================================
# TOOL NODE
# ============================================================

def tool_node(state: AgentState):

    print("\n========================================")
    print("LANGGRAPH → TOOL NODE")
    print("========================================")

    messages = state["messages"]

    last_message = messages[-1]

    tool_results = []

    total_tool_latency = state.get(
        "total_tool_latency",
        0.0
    )

    errors = state.get(
        "errors",
        []
    )

    for tool_call in last_message.tool_calls:

        tool_name = tool_call.function.name

        arguments = json.loads(
            tool_call.function.arguments
        )

        print("Tool selected :", tool_name)
        print("Arguments     :", arguments)

        tool_function = available_tools.get(
            tool_name
        )

        # Start tool timer
        tool_start = time.perf_counter()

        try:

            if tool_function is None:

                result = {
                    "success": False,
                    "message": "Unknown tool."
                }

            else:

                result = tool_function(
                    **arguments
                )

        except Exception as e:

            print(
                "\nTOOL ERROR:",
                str(e)
            )

            errors.append(
                f"{tool_name}: {str(e)}"
            )

            result = {
                "success": False,
                "message": "Tool execution failed."
            }

        # End tool timer
        tool_latency = (
            time.perf_counter()
            - tool_start
        )

        total_tool_latency += tool_latency

        print(
            "Tool latency  :",
            round(tool_latency, 4),
            "seconds"
        )

        print(
            "Tool result   :",
            result
        )

        tool_results.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": json.dumps(
                    result,
                    default=str
                )
            }
        )

    messages.extend(tool_results)

    return {
        "messages": messages,
        "total_tool_latency":
            total_tool_latency,
        "errors": errors
    }


# ============================================================
# ROUTING
# ============================================================

def route_after_agent(state: AgentState):

    messages = state["messages"]

    last_message = messages[-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return END


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph = StateGraph(AgentState)

graph.add_node(
    "agent",
    agent_node
)

graph.add_node(
    "tools",
    tool_node
)

graph.add_edge(
    START,
    "agent"
)

graph.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "tools": "tools",
        END: END
    }
)

graph.add_edge(
    "tools",
    "agent"
)

app = graph.compile()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # REQUEST TIMER
    # ========================================================

    request_start = time.perf_counter()

    user_query = input(
        "\nAsk something about an order/customer "
        "or company policy: "
    )

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

    # ========================================================
    # INITIAL STATE
    # ========================================================

    initial_state = {
        "user_query": user_query,

        "messages": [
            system_message,
            {
                "role": "user",
                "content": user_query
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

    print("\n========================================")
    print("LANGGRAPH AGENT STARTED")
    print("========================================")

    # ========================================================
    # RUN GRAPH
    # ========================================================

    result = app.invoke(
        initial_state
    )

    # ========================================================
    # TOTAL REQUEST LATENCY
    # ========================================================

    total_request_latency = (
        time.perf_counter()
        - request_start
    )

    final_message = result["messages"][-1]

    # ========================================================
    # FINAL ANSWER
    # ========================================================

    print("\n========================================")
    print("FINAL ANSWER")
    print("========================================")

    print(
        final_message.content
    )

    # ========================================================
    # PERFORMANCE SUMMARY
    # ========================================================

    print("\n========================================")
    print("PERFORMANCE SUMMARY")
    print("========================================")

    print(
        "LLM calls          :",
        result.get("llm_calls", 0)
    )

    print(
        "Input tokens       :",
        result.get(
            "total_input_tokens",
            0
        )
    )

    print(
        "Output tokens      :",
        result.get(
            "total_output_tokens",
            0
        )
    )

    print(
        "Total tokens       :",
        result.get(
            "total_tokens",
            0
        )
    )

    print(
        "Estimated cost     : $",
        round(
            result.get(
                "total_cost",
                0.0
            ),
            8
        )
    )

    print(
        "Total LLM latency  :",
        round(
            result.get(
                "total_llm_latency",
                0.0
            ),
            4
        ),
        "seconds"
    )

    print(
        "Total tool latency :",
        round(
            result.get(
                "total_tool_latency",
                0.0
            ),
            4
        ),
        "seconds"
    )

    print(
        "Total latency      :",
        round(
            total_request_latency,
            4
        ),
        "seconds"
    )

    print(
        "Errors             :",
        len(
            result.get(
                "errors",
                []
            )
        )
    )

    print("\n========================================")
    print("LANGGRAPH FLOW COMPLETED")
    print("========================================")
