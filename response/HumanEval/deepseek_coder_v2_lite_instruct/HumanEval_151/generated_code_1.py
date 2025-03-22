def method():
    input_list = [1, 3, 2, 0]  # Example input list
    output = 0
    
    for num in input_list:
        if isinstance(num, int) and num >= 0 and num % 2 != 0:
            output += num ** 2
    
    return output

# Test case
print(method())  # Should print 10