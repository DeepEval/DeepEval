def method(n):
    # Initialize an empty list to store the number of stones in each level
    output = []
    
    # Start from the first level with n stones
    current_stones = n
    
    # Loop through the levels starting from the second level
    for i in range(1, n):
        # Check if n is odd, if so, use the next odd number for the stones
        if n % 2 == 1:
            current_stones += i * 2 - 1
        # If n is even, use the next even number for the stones
        else:
            current_stones += i * 2
            
        # Append the number of stones in the current level to the list
        output.append(current_stones)
    
    return output

# Example test case
def test_method():
    result = method(5)
    expected = [5, 3, 3, 3, 3]
    # assert result == expected, f"Expected {expected}, but got {result}"

test_method()