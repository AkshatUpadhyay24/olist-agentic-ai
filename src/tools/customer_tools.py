import os
import pandas as pd


# Get the project root directory
# customer_tools.py is inside:
# D:\AGENTIC_AI_PROJECT\src\tools\
#
# So:
# tools -> src -> AGENTIC_AI_PROJECT

CURRENT_FILE = os.path.abspath(__file__)

TOOLS_DIRECTORY = os.path.dirname(CURRENT_FILE)

SRC_DIRECTORY = os.path.dirname(TOOLS_DIRECTORY)

PROJECT_ROOT = os.path.dirname(SRC_DIRECTORY)


# Build an absolute path to the orders dataset
ORDERS_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "olist_orders_dataset.csv"
)


# Load orders data
orders_df = pd.read_csv(ORDERS_PATH)


def get_customer_orders(customer_id: str) -> dict:
    """
    Get all orders belonging to a customer.
    """

    customer_orders = orders_df[
        orders_df["customer_id"] == customer_id
    ]

    if customer_orders.empty:
        return {
            "success": False,
            "customer_id": customer_id,
            "message": "No orders found for this customer."
        }

    orders = customer_orders[
        [
            "order_id",
            "order_status",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]
    ].to_dict(orient="records")

    return {
        "success": True,
        "customer_id": customer_id,
        "total_orders": len(orders),
        "orders": orders
    }