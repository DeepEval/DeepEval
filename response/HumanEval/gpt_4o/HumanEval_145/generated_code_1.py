def method(lst):
    # Function to calculate the sum of the digits of a number
    def sum_of_digits(n):
        return sum(int(digit) for digit in str(abs(n)))
    
    # Sort the list using the sum of digits as the primary key
    # The original index will naturally be used as a tiebreaker due to stable sorting
    sorted_lst = sorted(lst, key=sum_of_digits)
    
    return sorted_lst

# Test case for validation
input_list = [56, 23, 89, 12, 123, 45]
output = method(input_list)
print(output)  # Expected output: [12, 123, 23, 45, 56, 89]