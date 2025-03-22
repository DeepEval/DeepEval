def method(numbers):
    """
    This function takes an array of numbers as input and returns the number of elements in the array that are greater than 10 and both first and last digits of a number are odd.

    Parameters:
    numbers (list): A list of integers.

    Returns:
    int: The number of elements in the array that meet the condition.
    """
    
    # Initialize a counter variable to store the count of numbers that meet the condition
    count = 0
    
    # Iterate over each number in the input list
    for num in numbers:
        
        # Check if the number is greater than 10
        if num > 10:
            
            # Convert the number to a string to easily access its first and last digits
            str_num = str(num)
            
            # Check if both first and last digits of the number are odd
            if int(str_num[0]) % 2!= 0 and int(str_num[-1]) % 2!= 0:
                
                # If the condition is met, increment the counter
                count += 1
                
    # Return the count of numbers that meet the condition
    return count


# Test case for validation
numbers = [15, 20, 30, 51, 75, 101, 12, 18, 22, 39]
output = method(numbers)
print("Test case output:", output)