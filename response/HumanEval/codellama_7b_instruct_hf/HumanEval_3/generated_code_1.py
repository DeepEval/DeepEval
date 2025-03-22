import re

def method():
    operations = []
    balance = 0
    flag = False

    # Input operations from the user
    # while True:
    #     operation = input("Enter operation (deposit/withdrawal/exit): ")
    #     if operation == "exit":
    #         break
    #     amount = float(input("Enter amount: "))
    #     operations.append((operation, amount))
    
    # 修改
    operations = [("deposit", 100), ("withdrawal", 50), ("withdrawal", 70)]
    # Process operations and check for negative balance
    for operation, amount in operations:
        if operation == "deposit":
            balance += amount
        elif operation == "withdrawal":
            if balance - amount >= 0:
                balance -= amount
            else:
                flag = True
                break

    # Return True if negative balance is detected, False otherwise
    return flag


# Test case
operations = [("deposit", 100), ("withdrawal", 50), ("withdrawal", 70)]
result = method()
print(result)  # Output: True