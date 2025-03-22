def method():
    # Function to find the biggest even integer in the range [x, y] inclusive
    def find_biggest_even(x, y):
        # Initialize the result to -1
        result = -1
        
        # Iterate over the range from y down to x, incrementing by 2 to only consider even numbers
        for i in range(y, x - 1, -2):
            result = i  # Update the result if we find an even number
        
        return result
    
    # Example usage
    x = 10
    y = 20
    output = find_biggest_even(x, y)
    
    return output

# Test the function
print(method())  # Expected output: 18