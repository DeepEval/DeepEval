def method():
    def get_even_digits(start, end):
        even_digits = []
        for num in range(start, end + 1):
            for digit in str(num):
                if int(digit) % 2 == 0:
                    even_digits.append(int(digit))
        return sorted(even_digits)

    # Example usage:
    a = 10
    b = 30
    output = get_even_digits(a, b)
    return output

# Test case
a = 10
b = 30
expected_output = [0, 2, 4, 6, 8]
result = method()
print("Test Case:", result == expected_output)