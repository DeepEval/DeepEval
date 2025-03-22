def method():
    def parse_fraction(fraction):
        numerator, denominator = map(int, fraction.split('/'))
        return numerator, denominator

    def multiply_fractions(frac1, frac2):
        num1, den1 = parse_fraction(frac1)
        num2, den2 = parse_fraction(frac2)
        new_numerator = num1 * num2
        new_denominator = den1 * den2
        return new_numerator, new_denominator

    def is_whole_number(numerator, denominator):
        result = numerator / denominator
        return result.is_integer()

    x, n = "1/5", "5/1"
    numerator, denominator = multiply_fractions(x, n)
    output = is_whole_number(numerator, denominator)
    return output

# Test case
print(method())  # Expected output: True