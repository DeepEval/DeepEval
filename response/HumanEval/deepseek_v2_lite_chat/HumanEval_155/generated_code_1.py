def method(number):
    # Initialize counters for even and odd digits
    even_count = 0
    odd_count = 0
    
    # Convert the number to a string to iterate over each digit
    number_str = str(number)
    
    # Iterate over each character in the string
    for digit_str in number_str:
        # Convert the character back to an integer
        digit = int(digit_str)
        
        # Check if the digit is even or odd and increment the appropriate counter
        if digit % 2 == 0:
            even_count += 1
        else:
            odd_count += 1
    
    # Return the tuple containing the counts of even and odd digits
    return (even_count, odd_count)

# Example test case
def test_method():
    result = method(123456)
    print(result)
    # assert result == (2, 4), "Test case failed"
    # print("All test cases pass")

# Run the test case
test_method()