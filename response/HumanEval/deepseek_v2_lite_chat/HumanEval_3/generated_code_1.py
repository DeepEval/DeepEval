def method(operations):
    # Initialize balance as 0
    balance = 0
    
    # Iterate through each operation
    for operation in operations:
        if operation['type'] == 'deposit':
            # If deposit, add the amount to the balance
            balance += operation['amount']
        elif operation['type'] == 'withdrawal':
            # If withdrawal, subtract the amount from the balance
            # If the withdrawal would result in a negative balance, return True
            if balance - operation['amount'] < 0:
                return True
            balance -= operation['amount']
    
    # If no negative balance is detected, return False
    return False

# Example usage with a test case
operations = [
    {'type': 'deposit', 'amount': 100},
    {'type': 'withdrawal', 'amount': 50},
    {'type': 'deposit', 'amount': 200}
]

# Test the function with the example operations
output = method(operations)
print(output)  # This should print False

# Test case where there's a negative balance
operations = [
    {'type': 'withdrawal', 'amount': 120},
    {'type': 'deposit', 'amount': 50}
]
output = method(operations)
print(output)  # This should print True