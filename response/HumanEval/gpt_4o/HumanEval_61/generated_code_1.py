def method(brackets):
    # Counter to track the balance of brackets
    balance = 0
    
    for char in brackets:
        if char == '(':
            balance += 1
        elif char == ')':
            balance -= 1
        
        # If balance is negative, there's a closing bracket without an opening match
        if balance < 0:
            return False
    
    # If balance is not zero, some opening brackets are not closed
    return balance == 0

# Test case for validation
test_brackets = "(()())"
print(method(test_brackets))  # Should return True

test_brackets_unmatched = "(()"
print(method(test_brackets_unmatched))  # Should return False