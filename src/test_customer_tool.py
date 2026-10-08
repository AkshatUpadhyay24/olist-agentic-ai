from tools.customer_tools import get_customer_orders


customer_id = input("Enter customer ID: ")

result = get_customer_orders(customer_id)

print("\n" + "=" * 70)
print("CUSTOMER ORDERS")
print("=" * 70)

print(result)