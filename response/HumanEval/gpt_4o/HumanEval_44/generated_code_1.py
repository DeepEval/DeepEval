def method(x, base):
    # Check if base is valid
    if not (2 <= base < 10):
        raise ValueError("Base must be between 2 and 9, inclusive.")

    # Handle the case for zero
    if x == 0:
        return "0"

    # Result string
    result = ""

    # Convert the number to the desired base
    while x > 0:
        remainder = x % base
        result = str(remainder) + result
        x = x // base

    return result

# Test case for validation
def test_method():
    # Example: Convert number 25 to base 3
    x = 25
    base = 3
    output = method(x, base)
    print(f"The number {x} in base {base} is: {output}")

# Run the test
test_method()