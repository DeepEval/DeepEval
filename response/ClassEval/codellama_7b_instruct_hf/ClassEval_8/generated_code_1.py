import math

class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance

    def deposit(self, amount):
        if amount < 0:
            raise ValueError("Invalid amount")
        self.balance += amount
        return self.balance

    def withdraw(self, amount):
        if amount < 0:
            raise ValueError("Invalid amount")
        if self.balance - amount < 0:
            raise ValueError("Insufficient balance")
        self.balance -= amount
        return self.balance

    def view_balance(self):
        return self.balance

    def transfer(self, other_account, amount):
        if amount < 0:
            raise ValueError("Invalid amount")
        self.balance -= amount
        other_account.balance += amount
        return (self.balance, other_account.balance)

if __name__ == "__main__":
    account1 = BankAccount(1000)
    account2 = BankAccount(0)
    account1.deposit(300)
    account1.transfer(account2, 100)
    print(account1.balance) # 700
    print(account2.balance) # 100