def method():
    # Example list of integers
    numbers = [2, 3, 4, 5, 6, 7, 8, 9]
    
    # Initialize the sum of odd elements in even positions
    sum_of_odds = 0
    
    # Iterate through the list with index
    for index in range(len(numbers)):
        if index % 2 == 0 and numbers[index] % 2 != 0:
            sum_of_odds += numbers[index]
    
    # Return the final output
    return sum_of_odds

# Test case
output = method()
print("Sum of odd elements in even positions:", output)