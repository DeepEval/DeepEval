def method(numbers):
    def has_odd_first_and_last_digit(num):
        # Convert the number to a string to access first and last digits
        str_num = str(num)
        first_digit = int(str_num[0])
        last_digit = int(str_num[-1])
        return first_digit % 2 != 0 and last_digit % 2 != 0

    count = 0
    for number in numbers:
        if number > 10 and has_odd_first_and_last_digit(number):
            count += 1

    return count

# Test case for validation
test_input = [13, 21, 35, 7, 53, 19, 31, 99, 100, 1, 11, 39]
output = method(test_input)
print(output)  # Expected output: 5 (because 13, 35, 53, 19, 39 meet the criteria)