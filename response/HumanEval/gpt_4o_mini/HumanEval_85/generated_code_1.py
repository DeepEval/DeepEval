def method(lst):
    # Initialize a variable to hold the sum of even elements at odd indices
    total_sum = 0
    
    # Iterate through the list with index
    for index in range(len(lst)):
        # Check if the index is odd
        if index % 2 == 1:
            # Check if the element at this index is even
            if lst[index] % 2 == 0:
                total_sum += lst[index]
    
    return total_sum

# Test case for validation
test_lst = [1, 2, 3, 4, 5, 6, 7, 8]  # The even elements at odd indices are 2 (index 1) and 4 (index 3)
output = method(test_lst)
print(output)  # Expected output: 2 + 4 = 6