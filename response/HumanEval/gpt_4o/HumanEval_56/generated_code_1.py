def method(brackets):
    # Initialize a counter to keep track of the balance of brackets
    balance = 0
    
    # Iterate over each character in the input string
    for char in brackets:
        if char == '<':
            # Increment the balance for every opening bracket
            balance += 1
        elif char == '>':
            # Decrement the balance for every closing bracket
            balance -= 1
            # If balance becomes negative, it means there is an unmatched closing bracket
            if balance < 0:
                return False
    
    # If balance is zero at the end, all brackets are matched
    return balance == 0

# Test cases
print(method("<>"))         # Expected: True
print(method("<<>>"))       # Expected: True
print(method("<><>"))       # Expected: True
print(method("><"))         # Expected: False
print(method("<<>"))        # Expected: False
print(method("<><><>>"))    # Expected: False
print(method(""))           # Expected: True (empty string is trivially balanced)