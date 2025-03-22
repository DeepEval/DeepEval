def method():
    # Initialize the count of 7's to 0
    count = 0
    
    # Iterate through integers less than n that are divisible by 11 or 13
    for i in range(1, n):
        # Check if the number is divisible by 11 or 13
        if i % 11 == 0 or i % 13 == 0:
            # Convert the number to a string to iterate over its digits
            str_i = str(i)
            # Count the occurrences of the digit 7
            for digit in str_i:
                if digit == '7':
                    count += 1
    
    # Return the count of 7's
    return count

# Example usage:
# test case where n = 50
output = method()
print("Number of 7's in numbers less than n which are divisible by 11 or 13:", output)