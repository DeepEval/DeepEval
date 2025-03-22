def method():
    def fibfib(n):
        if n <= 2:
            return 0
        elif n == 3:
            return 1
        else:
            return fibfib(n-1) + fibfib(n-2) + fibfib(n-3)
    
    n = 6  # Example input
    output = fibfib(n)
    return output

# Test Case
result = method()
print(f"The {n}-th FibFib number is: {result}")