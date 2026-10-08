from tools.order_tools import get_order_details


# ---------------------------------------------------------
# TEST ORDER TOOL
# ---------------------------------------------------------

order_id = input("Enter an order ID: ").strip()

result = get_order_details(order_id)

print("\n" + "=" * 70)
print("ORDER TOOL RESULT")
print("=" * 70)

print(result)