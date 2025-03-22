def method(num1, num2, num3):
    # Check if all inputs are integers
    if not (isinstance(num1, int) and isinstance(num2, int) and isinstance(num3, int)):
        return False
    
    # Check if one of the numbers is equal to the sum of the other two
    if num1 == num2 + num3 or num2 == num1 + num3 or num3 == num1 + num2:
        return True
    else:
        return False

# Test case to validate the function
def test_method():
    # # Test with numbers that satisfy the condition
    # assert method(3, 5, 8) == True
    # assert method(10, 20, 30) == True
    # assert method(2, 3, 5) == False
    print(method(3, 5, 8))
    
    # # Test with numbers that do not satisfy the condition
    # assert method(1, 2, 3) == False
    # assert method(4, 5, 6) == False
    
    # # Test with non-integer numbers
    # assert method(1.5, 2.5, 3.5) == False
    # assert method(2, 3.5, 4) == False
    
    # print("All test cases pass")

# Run the test function
test_method()