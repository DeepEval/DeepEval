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
        if amount < 0:
            raise ValueError("Invalid amount")
        if amount > self.balance:
            raise ValueError("Insufficient balance.")
        self.withdraw(amount)
        other_account.deposit(amount)
        return self.balance, other_account.balance

# Test cases for the BankAccount class
if __name__ == "__main__":
    # Test deposit
    account1 = BankAccount()
    print("Initial Balance:", account1.view_balance())  # Output: 0
    print("Depositing 1000:", account1.deposit(1000))    # Output: 1000

    # Test withdraw
    print("Withdrawing 300:", account1.withdraw(300))    # Output: 700
    try:
        print("Withdrawing 800:", account1.withdraw(800))  # Should raise ValueError
    except ValueError as e:
        print(e)  # Output: Insufficient balance.

    # Test view_balance
    print("Current Balance:", account1.view_balance())     # Output: 700

    # Test transfer
    account2 = BankAccount()
    print("Transferring 200 to account2...")
    print("After Transfer:", account1.transfer(account2, 200))  # Output: (500, 200)
    print("Account1 Balance:", account1.view_balance())           # Output: 500
    print("Account2 Balance:", account2.view_balance())           # Output: 200

    # Additional tests for invalid cases
    try:
        account1.withdraw(-100)  # Should raise ValueError
    except ValueError as e:
        print(e)  # Output: Invalid amount

    try:
        account1.deposit(-50)  # Should raise ValueError
    except ValueError as e:
        print(e)  # Output: Invalid amount