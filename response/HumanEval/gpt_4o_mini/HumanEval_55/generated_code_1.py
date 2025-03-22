def method(n):
    if n < 0:
        raise ValueError("Input should be a non-negative integer.")
    elif n == 0:
        return 0
    elif n == 1:
        return 1

    # Initialize the first two Fibonacci numbers
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b  # Update the values for the next Fibonacci number
    return b

# Test case
if __name__ == "__main__":
    # Example test case
    n = 10  # Change this value to test different cases
    output = method(n)
    print(f"The {n}-th Fibonacci number is: {output}")