# Programming for AI LAB - FALL 2024
# Lab - 5

class Account:
    def __init__(self):
        self.__account_no = ""
        self.__account_bal = ""
        self.__security_code = ""
    def initialize_account(self, account_no, account_bal, security_code):
        self.__account_no = account_no
        self.__account_bal = account_bal
        self.__security_code = security_code
    def print_account_info(self):
        print("Account Number:", self.__account_no)
        print("Account Balance:", self.__account_bal)
        print("Security Code:", self.__security_code)

a = Account()
a.initialize_account("579562597", 1000.0, "2036")
a.print_account_info()
