def display_menu():
    print("\nCurrency Converter")
    print("1. Euro (EUR)")
    print("2. Dollar (USD)")
    print("3. Pakistani Rupee (PKR)")
    print("4. Indian Rupee (INR)")
    print("5. Yen (JPY)")
    print("6. Exit")


def get_exchange_rate(from_currency, to_currency):
    rates = {
        'EUR': {'USD': 1.1, 'PKR': 305.0, 'INR': 90.0, 'JPY': 150.0},
        'USD': {'EUR': 0.91, 'PKR': 275.0, 'INR': 82.0, 'JPY': 137.0},
        'PKR': {'EUR': 0.0033, 'USD': 0.0036, 'INR': 0.30, 'JPY': 0.50},
        'INR': {'EUR': 0.011, 'USD': 0.012, 'PKR': 3.30, 'JPY': 1.7},
        'JPY': {'EUR': 0.0067, 'USD': 0.0073, 'PKR': 2.0, 'INR': 0.59}
    }
    return rates[from_currency][to_currency]


def convert_currency(amount, from_currency, to_currency):
    exchange_rate = get_exchange_rate(from_currency, to_currency)
    return amount * exchange_rate


while True:
    display_menu()

    choice = input("Enter your choice (1-6): ")

    if choice == '6':
        print("Exiting...")
        break

    currency_codes = {
        '1': 'EUR',
        '2': 'USD',
        '3': 'PKR',
        '4': 'INR',
        '5': 'JPY'
    }

    if choice not in currency_codes:
        print("Invalid choice. Please try again.")
        continue

    from_currency = currency_codes[choice]
    amount = float(input(f"Enter the amount in {from_currency}: "))

    print("Convert to:")
    for key, value in currency_codes.items():
        if value != from_currency:
            print(f"{key}. {value}")

    to_choice = input("Enter your choice (1-5): ")

    if to_choice not in currency_codes or to_choice == choice:
        print("Invalid choice or same currency. Please try again.")
        continue

    to_currency = currency_codes[to_choice]
    converted_amount = convert_currency(amount, from_currency, to_currency)

    print(f"{amount} {from_currency} is equal to {converted_amount:.2f} {to_currency}")

