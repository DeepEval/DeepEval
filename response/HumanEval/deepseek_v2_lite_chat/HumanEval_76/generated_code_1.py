from math import log

def is_simple_power(x, n):
    # Check if x can be represented as n raised to an integer power
    return log(x, n).is_integer()

# Test case to validate the function
def test_is_simple_power():
    # assert is_simple_power(8, 2) == True
    # assert is_simple_power(27, 3) == True
    # assert is_simple_power(64, 4) == True
    # assert is_simple_power(1, 1) == True
    # assert is_simple_power(1024, 2) == False
    # assert is_simple_power(125, 5) == False
    # assert is_simple_power(0, 5) == False
    # assert is_simple_power(1, 0) == True
    print(is_simple_power(0, 5))

# Example usage:
# print(method(8, 2))  # Should return True
# print(method(27, 3))  # Should return True
# print(method(64, 4))  # Should return True
# print(method(1, 1))   # Should return True
# print(method(1024, 2))  # Should return False
# print(method(125, 5))  # Should return False
# print(method(0, 5))  # Should return False
# print(method(1, 0))  # Should return True

if __name__ == "__main__":
    test_is_simple_power()