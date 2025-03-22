def method(n):
    def is_prime(num):
        if num <= 1:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    primes = [i for i in range(2, n) if is_prime(i)]

    return primes

# Example test case for validation
test_n = 10
output = method(test_n)
print(output)  # Expected output: [2, 3, 5, 7]