def method(num):
    # Convert the integer to its string representation
    str_num = str(num)
    
    # Initialize counters for even and odd digits
    even_count = 0
    odd_count = 0
    
    # Loop through each character in the string representation
    for char in str_num:
        # Check if the character is a digit
        if char.isdigit():
            digit = int(char)
            # Increment the even or odd counter based on the digit
            if digit % 2 == 0:
                even_count += 1
            else:
                odd_count += 1

    # Return the counts as a tuple
    return (even_count, odd_count)

# Test case
test_number = 1234567890
output = method(test_number)
print(f'For the number {test_number}, the counts of even and odd digits are: {output}')