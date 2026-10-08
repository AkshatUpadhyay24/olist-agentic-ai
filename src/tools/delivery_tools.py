import os
import pandas as pd


# Get the project root directory
# delivery_tools.py is inside:
# D:\AGENTIC_AI_PROJECT\src\tools\
#
# So:
# tools -> src -> AGENTIC_AI_PROJECT

CURRENT_FILE = os.path.abspath(__file__)

TOOLS_DIRECTORY = os.path.dirname(CURRENT_FILE)

SRC_DIRECTORY = os.path.dirname(TOOLS_DIRECTORY)

PROJECT_ROOT = os.path.dirname(SRC_DIRECTORY)


# Build absolute path to orders dataset
ORDERS_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "olist_orders_dataset.csv"
)


# Load orders data
orders_df = pd.read_csv(ORDERS_PATH)


def calculate_delivery_delay(order_id: str) -> dict:
    """
    Calculate delivery delay for a specific order.
    """

    order = orders_df[
        orders_df["order_id"] == order_id
    ]

    if order.empty:
        return {
            "success": False,
            "order_id": order_id,
            "message": "Order not found."
        }

    order = order.iloc[0]

    delivered_date = order[
        "order_delivered_customer_date"
    ]

    estimated_date = order[
        "order_estimated_delivery_date"
    ]

    # Check whether both dates are available
    if pd.isna(delivered_date):
        return {
            "success": True,
            "order_id": order_id,
            "delivered": False,
            "message": "Order has not been delivered yet."
        }

    if pd.isna(estimated_date):
        return {
            "success": True,
            "order_id": order_id,
            "delivered": True,
            "message": (
                "Order was delivered, "
                "but estimated delivery date "
                "is not available."
            )
        }

    delivered_date = pd.to_datetime(
        delivered_date
    )

    estimated_date = pd.to_datetime(
        estimated_date
    )

    delay_days = (
        delivered_date - estimated_date
    ).days

    return {
        "success": True,
        "order_id": order_id,
        "delivered": True,
        "delivered_date": str(
            delivered_date
        ),
        "estimated_delivery_date": str(
            estimated_date
        ),
        "delay_days": delay_days,
        "delayed": delay_days > 0
    }