class Account:
    def __init__(self):
        self.__account_no = None
        self.__account_bal = None
        self.__security_code = None

    def initialize_and_print(self, account_no, account_bal, security_code):
        """Initialize the account details and print them."""
        self.__account_no = account_no
        self.__account_bal = account_bal
        self.__security_code = security_code

        print(f"Account Number: {self.__account_no}")
        print(f"Account Balance: ${self.__account_bal:.2f}")
        print(f"Security Code: {self.__security_code}")


if __name__ == "__main__":
    account = Account()

    account.initialize_and_print(account_no="1234567890", account_bal=1500.75, security_code="XYZ123")
