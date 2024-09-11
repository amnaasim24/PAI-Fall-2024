class Restaurant:
    def __init__(self):
        self.menu_items = {}
        self.book_table = {}
        self.customer_orders = {}
    def add_item_to_menu(self, item, price):
        self.menu_items[item] = price
        print("Added " + item + " to the menu")
    def book_tables(self, table_number, customer_name):
        self.book_table[table_number] = customer_name
        print("Table " + str(table_number) + " booked")
    def customer_order(self, table_number, order):
        if table_number in self.book_table:
            self.customer_orders[table_number] = order
            print("Order taken for table " + str(table_number))
        else:
            print("Table not booked")
    def print_menu(self):
        print("Menu:")
        for item in self.menu_items:
            print(item)
    def print_table_reservations(self):
        print("Table Reservations:")
        for table_number in self.book_table:
            print("Table " + str(table_number))
    def print_customer_orders(self):
        print("Customer Orders:")
        for table_number in self.customer_orders:
            print("Table " + str(table_number))

restaurant = Restaurant()

restaurant.add_item_to_menu("Burger", 10.99)
restaurant.add_item_to_menu("Pizza", 14.99)
restaurant.add_item_to_menu("Salad", 8.99)

restaurant.book_tables(1, "John")
restaurant.book_tables(2, "Jane")
restaurant.book_tables(3, "Bob")

restaurant.customer_order(1, "Burger and Fries")
restaurant.customer_order(2, "Pizza and Salad")
restaurant.customer_order(3, "Salad and Drink")

restaurant.print_menu()
restaurant.print_table_reservations()
restaurant.print_customer_orders()
