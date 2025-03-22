def is_odd(digit):
    return digit in [1, 3, 5, 7, 9]

def get_first_digit(number):
    while number >= 10:
        number //= 10
    return number

def get_last_digit(number):
    return number % 10

def method(numbers):
    count = 0
    for number in numbers:
        if number > 10:
            first_digit = get_first_digit(number)
            last_digit = get_last_digit(number)
            if is_odd(first_digit) and is_odd(last_digit):
                count += 1
    return count

# Test case
numbers = [123, 456, 789, 101, 234, 567, 890, 12345, 67890]
output = method(numbers)
print("Number of elements greater than 10 with both first and last digits as odd:", output)