class BankAccount:
    """
    This is a class as a bank account system, which supports deposit money, withdraw money, view balance, and transfer money.
    """

    def __init__(self, balance=0):
        """
        Initializes a bank account object with an attribute balance, default value is 0.
        """
        self.balance = balance

    def deposit(self, amount):
        """
        Deposits a certain amount into the account, increasing the account balance, return the current account balance.
        If amount is negative, raise a ValueError("Invalid amount").
        :param amount: int
        """
        if amount < 0:
            raise ValueError("Invalid amount")
        self.balance += amount
        return self.balance

    def withdraw(self, amount):
        """
        Withdraws a certain amount from the account, decreasing the account balance, return the current account balance.
        If amount is negative, raise a ValueError("Invalid amount").
        If the withdrawal amount is greater than the account balance, raise a ValueError("Insufficient balance.").
        :param amount: int
        """
        if amount < 0:
            raise ValueError("Invalid amount")
        if amount > self.balance:
            raise ValueError("Insufficient balance.")
        self.balance -= amount
        return self.balance

    def view_balance(self):
        """
        Return the account balance.
        """
        return self.balance

    def transfer(self, other_account, amount):
        """
        Transfers a certain amount from the current account to another account.
        :param other_account: BankAccount
        :param amount: int
        """
        self.withdraw(amount)
        other_account.deposit(amount)


# Test cases
if __name__ == "__main__":
    account1 = BankAccount()
    account2 = BankAccount()
    
    # Test deposit
    try:
        print(account1.deposit(1000))  # Expected output: 1000
    except ValueError as e:
        print(e)
    
    # Test withdraw
    try:
        print(account1.withdraw(300))  # Expected output: 700
    except ValueError as e:
        print(e)
    
    # Test view_balance
    print(account1.view_balance())  # Expected output: 700
    
    # Test transfer
    try:
        account1.transfer(account2, 300)
        print(f"account1.balance = {account1.view_balance()} account2.balance = {account2.view_balance()}")
        # Expected output: account1.balance = 700 account2.balance = 300
    except ValueError as e:
        print(e)