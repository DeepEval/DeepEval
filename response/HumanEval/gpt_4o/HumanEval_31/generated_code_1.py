def method(number):
    if number <= 1:
        return False
    if number <= 3:
        return True
    if number % 2 == 0 or number % 3 == 0:
        return False
    i = 5
    while i * i <= number:
        if number % i == 0 or number % (i + 2) == 0:
            return False
        i += 6
    return True

# Example test case
print(method(5))  # Expected output: True, because 5 is a prime number
print(method(4))  # Expected output: False, because 4 is not a prime number