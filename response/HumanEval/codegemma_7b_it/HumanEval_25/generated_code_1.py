import math

def method():
    def get_prime_factorization(num):
        prime_factors = {}

        for i in range(2, int(math.sqrt(num)) + 1):
            while num % i == 0:
                prime_factors[i] = prime_factors.get(i, 0) + 1
                num //= i

        if num > 1:
            prime_factors[num] = 1

        return prime_factors

    def get_factorization(num):
        factorization = []

        for factor, exponent in get_prime_factorization(num).items():
            factorization.extend([factor] * exponent)

        return factorization

    num = 120  # Test case input
    factorization = get_factorization(num)
    output = factorization

    return output

print(method())  # Output: [2, 2, 2, 3, 5]