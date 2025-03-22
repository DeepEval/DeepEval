from fractions import Fraction

def method():
    def is_whole_number(x):
        return x % 1 == 0

    def convert_to_decimal(fraction):
        return float(Fraction(fraction))

    def multiply_and_check(x, n):
        decimal_x = convert_to_decimal(x)
        decimal_n = convert_to_decimal(n)
        product = decimal_x * decimal_n
        return is_whole_number(product)

    x = "1/5"
    n = "5/1"
    return multiply_and_check(x, n)

output = method()
print(output)  # Expected output: True