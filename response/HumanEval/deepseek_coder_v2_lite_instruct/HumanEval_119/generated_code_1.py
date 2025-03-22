def is_balanced(s):
    balance = 0
    for char in s:
        if char == '(':
            balance += 1
        else:
            balance -= 1
        if balance < 0:
            return False
    return balance == 0

def method():
    str1 = "()"
    str2 = "()"
    
    # Check both possible concatenations
    if is_balanced(str1 + str2) or is_balanced(str2 + str1):
        output = "Yes"
    else:
        output = "No"
    
    return output

# Test case
print(method())  # Expected output: "Yes"