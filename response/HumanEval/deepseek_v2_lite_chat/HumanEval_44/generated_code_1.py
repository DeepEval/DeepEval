def method():
    # Mapping of digits to their corresponding characters
    digit_to_char = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
        5: '5', 6: '6', 7: '7', 8: '8', 9: '9'
    }

    def convert_base(x, base):
        # Helper function to convert from base 10 to base 36
        def from_decimal(num):
            digits = []
            while num:
                num, digit = divmod(num, base)
                digits.insert(0, digit_to_char[int(digit)])
            return ''.join(digits)

        # Convert the number to the given base
        return from_decimal(x)

    # Example usage
    output = convert_base(1024, 2)  # Convert from base 10 to base 2
    return output

# Test case
print(method())