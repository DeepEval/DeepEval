def method():
    lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    output = 0
    
    for i in range(1, len(lst), 2):  # Iterate over odd indices
        if lst[i] % 2 == 0:  # Check if the element at the odd index is even
            output += lst[i]
    
    return output

# Test case
print(method())  # Expected output: 12 (even elements at odd indices are 2, 4, 6, 8, 10, sum is 30)