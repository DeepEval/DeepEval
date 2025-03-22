import sympy

def is_prime(num):
    return sympy.isprime(num)

def generate_fibonacci(n):
    fib_sequence = [0, 1]
    while len(fib_sequence) <= n:
        next_fib = fib_sequence[-1] + fib_sequence[-2]
        fib_sequence.append(next_fib)
    return fib_sequence

def method():
    n = 10  # Example n, you can change this to test other values
    fib_sequence = generate_fibonacci(n*10)  # Generate more Fibonacci numbers than needed
    prime_fib_numbers = [num for num in fib_sequence if is_prime(num)]
    
    if len(prime_fib_numbers) >= n:
        output = prime_fib_numbers[n-1]
    else:
        output = "Not enough prime Fibonacci numbers found"
    
    return output

# Test case
print(method())  # This should print the 10th prime Fibonacci number