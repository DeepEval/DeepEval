def count_nums(numbers):
    def sum_of_digits(n):
        s = str(n)
        if s[0] == '-':
            return -int(s[1]) + sum(int(digit) for digit in s[2:])
        else:
            return sum(int(digit) for digit in s)
    
    count = 0
    for number in numbers:
        if sum_of_digits(number) > 0:
            count += 1
    return count

def method():
    # Test case
    numbers = [123, -45, 67, -8, 9]
    output = count_nums(numbers)
    return output

# Running the method to get the output
output = method()
print(output)  # Expected output: 3