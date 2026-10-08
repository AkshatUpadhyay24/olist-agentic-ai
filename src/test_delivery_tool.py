from tools.delivery_tools import calculate_delivery_delay


order_id = input("Enter order ID: ")

result = calculate_delivery_delay(order_id)

print("\n" + "=" * 70)
print("DELIVERY DELAY RESULT")
print("=" * 70)

print(result)