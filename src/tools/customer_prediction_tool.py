import os
import pandas as pd


CURRENT_FILE = os.path.abspath(__file__)
TOOLS_DIRECTORY = os.path.dirname(CURRENT_FILE)
SRC_DIRECTORY = os.path.dirname(TOOLS_DIRECTORY)
PROJECT_ROOT = os.path.dirname(SRC_DIRECTORY)

ORDERS_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "olist_orders_dataset.csv"
)

orders_df = pd.read_csv(ORDERS_PATH)


def predict_customer_repeat_order(customer_id: str) -> dict:
    """
    Predict whether a customer is likely to place
    another order based on their historical order count.

    This is a simple rule-based ML placeholder
    for learning the Agent -> ML Tool architecture.
    """

    customer_orders = orders_df[
        orders_df["customer_id"] == customer_id
    ]

    if customer_orders.empty:
        return {
            "success": False,
            "customer_id": customer_id,
            "message": "Customer not found."
        }

    order_count = len(customer_orders)

    if order_count >= 2:
        prediction = "HIGH"
        probability = 0.80
    else:
        prediction = "LOW"
        probability = 0.30

    return {
        "success": True,
        "customer_id": customer_id,
        "historical_orders": order_count,
        "repeat_order_prediction": prediction,
        "probability": probability
    }


if __name__ == "__main__":

    customer_id = orders_df.iloc[0]["customer_id"]

    result = predict_customer_repeat_order(
        customer_id
    )

    print(result)