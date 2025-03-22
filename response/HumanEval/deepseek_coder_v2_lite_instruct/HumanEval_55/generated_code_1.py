def method():
    def fibonacci(n):
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b
    
    # Example usage:
    n = 10  # This is the n-th Fibonacci number you want to find
    output = fibonacci(n)
    return output

# Test case
print(method())  # Should print the 10th Fibonacci number