def method():
    # Import necessary packages if needed
    # For this task, no external packages are required
    
    def sum_of_digits(n):
        return sum(int(digit) for digit in str(n))
    
    input_list = [123, 456, 789, 101, 234]
    sorted_list = sorted(input_list, key=lambda x: (sum_of_digits(x), input_list.index(x)))
    
    output = sorted_list
    return output

# Test case
print(method())  # Expected output: [101, 123, 234, 456, 789]