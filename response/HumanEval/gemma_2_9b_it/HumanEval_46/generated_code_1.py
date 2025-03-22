def method():
    def fib4(n):
        if n <= 3:
            return [0, 0, 2, 0][n]
        
        a, b, c, d = 0, 0, 2, 0  
        for i in range(4, n + 1):
            next_fib = a + b + c + d
            a, b, c, d = b, c, d, next_fib
        return d

    n = 10  # Example input
    output = fib4(n)
    return output

# Test case
print(method())  # This will print the 10th Fib4 number