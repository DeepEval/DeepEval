def method():
    # Define the string of brackets
    # brackets = input()
    brackets = "<<<<>>><<"
    
    # Initialize an empty stack
    stack = []
    
    # Iterate through each bracket in the string
    for bracket in brackets:
        if bracket == "<":
            # Push the opening bracket onto the stack
            stack.append(bracket)
        elif stack:
            # If the stack is not empty, pop the opening bracket from the stack
            stack.pop()
        else:
            # If the stack is empty, return False
            return False
    
    # If all opening brackets have their corresponding closing brackets, return True
    return len(stack) == 0

# Test case to validate the function
def test_method():
    # assert method() == True
    # assert method() == True
    # assert method() == True
    # assert method() == False  # This should return False because the first "<" has no corresponding ">"
    # assert method() == False  # This should return False because the second "<" has no corresponding ">"
    print(method())

# Run the test case
test_method()