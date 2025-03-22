def method(l):
    # Importing required packages
    from statistics import median
    
    # Calculate the median using the statistics module
    output = median(l)
    
    return output

# Test case for validation
if __name__ == "__main__":
    test_list = [3, 1, 4, 1, 5, 9, 2]
    print("The median of the list is:", method(test_list))  # Expected output: 3