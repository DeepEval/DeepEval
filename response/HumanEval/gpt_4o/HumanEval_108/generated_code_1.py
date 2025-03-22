def count_nums(arr):
    def sum_of_digits(n):
        n_str = str(n)
        total = 0
        for i, char in enumerate(n_str):
            if char == '-':
                continue
            digit = int(char)
            if i == 0 and n_str[0] == '-':
                digit = -digit
            total += digit
        return total
    
    count = 0
    for num in arr:
        if sum_of_digits(num) > 0:
            count += 1
    
    return count

def method():
    # Test the function with a test case
    test_case = [-123, 456, -789, 0, 12, -3, 100]
    output = count_nums(test_case)
    return output

# Example of running the method to get the output
print(method())  # Expected output: 4