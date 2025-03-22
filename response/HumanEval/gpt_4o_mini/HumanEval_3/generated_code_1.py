def method(operations):
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False

# Test case
operations = [100, -50, -30, 20, -100]  # This should return True as the balance goes below zero
output = method(operations)
print(output)  # Output: True

operations2 = [100, 50, 20, -30, -10]  # This should return False as the balance never goes below zero
output2 = method(operations2)
print(output2)  # Output: False