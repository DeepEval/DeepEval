class BankAccount:
    """
    This is a class for a bank account system, which supports deposit money, withdraw money, view balance, and transfer money.
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
        if amount < 0:
            raise ValueError("Invalid amount")
        if amount > self.balance:
            raise ValueError("Insufficient balance.")
        self.balance -= amount
        other_account.deposit(amount)
        return self.balance, other_account.balance

if __name__ == "__main__":
    # Test case for deposit
    account = BankAccount()
    print("Deposit Test")
    print(account.deposit(100))  # Expected output: 100

    # Test case for withdraw
    print("\nWithdraw Test")
    print(account.withdraw(50))  # Expected output: 50

    # Test case for view_balance
    print("\nView Balance Test")
    print(account.view_balance())  # Expected output: 50

    # Test case for transfer
    print("\nTransfer Test")
    account2 = BankAccount()
    account.deposit(200)
    print(account.transfer(account2, 100))  # Expected output: (150, 100)