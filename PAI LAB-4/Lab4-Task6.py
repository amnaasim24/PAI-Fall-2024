# Programming for AI LAB - FALL 2024
# Lab - 4

class BankAccount:
    def __init__(self, account_number, customer_name, date_of_opening, balance=0.0):
        self.account_number = account_number
        self.customer_name = customer_name
        self.date_of_opening = date_of_opening
        self.balance = balance

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
        else:
            print("Invalid deposit amount.")

    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
        else:
            print("Invalid withdrawal amount.")

    def check_balance(self):
        print("Account balance:", self.balance)



account = BankAccount("1234567890", "Amna Asim", "2022-01-01")
account.deposit(100.0)
account.withdraw(50.0)
account.check_balance()
