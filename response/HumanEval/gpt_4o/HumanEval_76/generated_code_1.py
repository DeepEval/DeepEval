import math

def method(x, n):
    if n == 0:
        return x == 1
    if n == 1:
        return x == 1
    if x < 1:
        return False
    
    # Calculate the logarithm of x with base n
    try:
        log_result = math.log(x, n)
        # Check if log_result is an integer
        return math.isclose(log_result, round(log_result), rel_tol=1e-9)
    except ValueError:
        # This handles the case where n is less than or equal to 0 and x is positive
        return False

# Test cases for validation
print(method(8, 2))  # True: 2^3 = 8
print(method(9, 3))  # True: 3^2 = 9
print(method(10, 2)) # False: 10 is not a power of 2
print(method(1, 5))  # True: 5^0 = 1
print(method(0, 0))  # False: 0^p is undefined for positive p
print(method(1, 0))  # True: 0^0 is conventionally 1
print(method(1, 1))  # True: 1^0 = 1