
from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# DATA LOCATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "raw"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

orders_df = pd.read_csv(
    DATA_DIR / "olist_orders_dataset.csv"
)

customers_df = pd.read_csv(
    DATA_DIR / "olist_customers_dataset.csv"
)

items_df = pd.read_csv(
    DATA_DIR / "olist_order_items_dataset.csv"
)

payments_df = pd.read_csv(
    DATA_DIR / "olist_order_payments_dataset.csv"
)

reviews_df = pd.read_csv(
    DATA_DIR / "olist_order_reviews_dataset.csv"
)


# ---------------------------------------------------------
# TOOL
# ---------------------------------------------------------

def get_order_details(order_id: str) -> dict:
    """
    Retrieve complete information about an order.

    Parameters
    ----------
    order_id : str
        Unique Olist order ID.

    Returns
    -------
    dict
        Structured order information.
    """

    # 1. Find order
    order = orders_df[
        orders_df["order_id"] == order_id
    ]

    if order.empty:
        return {
            "success": False,
            "error": f"Order '{order_id}' not found."
        }

    order = order.iloc[0]

    # 2. Find customer
    customer = customers_df[
        customers_df["customer_id"] == order["customer_id"]
    ]

    customer_data = {}

    if not customer.empty:
        customer = customer.iloc[0]

        customer_data = {
            "customer_id": customer["customer_id"],
            "city": customer["customer_city"],
            "state": customer["customer_state"]
        }

    # 3. Find order items
    items = items_df[
        items_df["order_id"] == order_id
    ]

    item_data = []

    for _, item in items.iterrows():

        item_data.append({
            "product_id": item["product_id"],
            "seller_id": item["seller_id"],
            "price": float(item["price"]),
            "freight_value": float(item["freight_value"])
        })

    # 4. Find payments
    payments = payments_df[
        payments_df["order_id"] == order_id
    ]

    payment_data = []

    for _, payment in payments.iterrows():

        payment_data.append({
            "payment_type": payment["payment_type"],
            "installments": int(
                payment["payment_installments"]
            ),
            "payment_value": float(
                payment["payment_value"]
            )
        })

    # 5. Find review
    review = reviews_df[
        reviews_df["order_id"] == order_id
    ]

    review_data = []

    for _, r in review.iterrows():

        review_data.append({
            "review_score": int(r["review_score"]),
            "comment": r["review_comment_message"]
        })

    # 6. Build final structured response
    result = {

        "success": True,

        "order_id": order_id,

        "order": {
            "status": order["order_status"],
            "purchase_timestamp":
                order["order_purchase_timestamp"],
            "approved_at":
                order["order_approved_at"],
            "delivered_to_carrier":
                order["order_delivered_carrier_date"],
            "delivered_to_customer":
                order["order_delivered_customer_date"],
            "estimated_delivery":
                order["order_estimated_delivery_date"]
        },

        "customer": customer_data,

        "items": item_data,

        "payments": payment_data,

        "reviews": review_data
    }

    return result