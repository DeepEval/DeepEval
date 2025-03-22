def method():
    # Define the array of non-negative integers
    arr = [7, 6, 15, 8]
    
    # Define the helper function to count the number of ones in the binary representation
    def count_ones(n):
        return bin(n).count('1')
    
    # Sort the array based on the number of ones in the binary representation
    # If two numbers have the same number of ones, sort by decimal value
    sorted_arr = sorted(arr, key=lambda x: (count_ones(x), x))
    
    # Return the sorted array
    return sorted_arr

# Test case
output = method()
print(output)  # Expected output: [6, 7, 8, 15]