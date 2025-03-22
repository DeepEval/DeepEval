def method(operations):
    balance = 0

    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    
    return False

# Test case for validation
# This test case includes a series of operations: [+100, -150, +50, +200, -300]
# The balance goes negative after the second operation (-150), so the expected output is True.
test_operations = [100, -150, 50, 200, -300]
output = method(test_operations)
print(output)  # Expected output: True