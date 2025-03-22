def method():
    import re

    def get_even_digits(start, end):
        # Convert the start and end to strings for easier comparison
        start_str, end_str = str(start), str(end)
        even_digits = []

        # Check each digit in the range
        for digit in range(int(start_str), int(end_str) + 1):
            # Check if the digit is even
            if int(digit) % 2 == 0:
                even_digits.append(digit)

        return even_digits

    # Example usage:
    start = 2
    end = 10
    output = get_even_digits(start, end)
    return output

# Test case
print(method())