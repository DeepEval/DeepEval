def method():
    def count_even_odd_digits(n):
        even_count = 0
        odd_count = 0
        for digit in str(n):
            if int(digit) % 2 == 0:
                even_count += 1
            else:
                odd_count += 1
        return (even_count, odd_count)

    # Example usage:
    number = 123456
    output = count_even_odd_digits(number)
    return output

# Test case
test_number = 123456
expected_output = (3, 3)
result = method()

# assert result == expected_output, f"Test failed! Expected {expected_output}, but got {result}"
# print("Test passed!")