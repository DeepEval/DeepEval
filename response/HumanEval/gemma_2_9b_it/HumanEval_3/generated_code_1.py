def method(operations):
    balance = 0
    for operation in operations:
        if operation[0] == 'Deposit':
            balance += operation[1]
        elif operation[0] == 'Withdrawal':
            if balance >= operation[1]:
                balance -= operation[1]
            else:
                return True  
    return False

# Test case
operations = [
    ['Deposit', 1000],
    ['Withdrawal', 200],
    ['Deposit', 500],
    ['Withdrawal', 700],
]

output = method(operations)
print(f"Output: {output}")  # Output: False 


operations = [
    ['Deposit', 1000],
    ['Withdrawal', 1200],
]

output = method(operations)
print(f"Output: {output}")  # Output: True