from math import sqrt

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def prime_factors(n):
    factors = []
    # Divide n by 2 to check if it's divisible
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    # n must be odd at this point, thus a skip of 2 can be used
    for i in range(3, int(sqrt(n)) + 1, 2):
        while n % i == 0:
            factors.append(i)
            n //= i
    # This condition is to handle the case when n is a prime number greater than 2
    if n > 2:
        factors.append(n)
    return factors

def method():
    # Input number
    # num = int(input("Enter a number to find its prime factors: "))
    num = 120
    
    # Find prime factors
    prime_factors_list = prime_factors(num)
    
    # Sort factors in ascending order and count frequency
    factors_with_frequency = {}
    for factor in prime_factors_list:
        if factor in factors_with_frequency:
            factors_with_frequency[factor] += 1
        else:
            factors_with_frequency[factor] = 1
    
    # Reconstruct the product of all factors
    product = 1
    for factor, frequency in factors_with_frequency.items():
        product *= factor ** frequency
    
    # Return the product and the list of factors with their frequencies
    output = (product, factors_with_frequency)
    return output

# Test case
result = method()
print("Test Case:", result)

# Additional test cases can be added here
additional_test_cases = [(1, {1: 1}), (2, {2: 1}), (4, {2: 2}), (8, {2: 3}),
                          (10, {1: 1, 2: 1, 5: 1}), (18, {2: 1, 3: 2}),
                          (36, {2: 2, 3: 2}), (120, {2: 2, 3: 1, 5: 1})]

for test_case in additional_test_cases:
    result = method(*test_case)
    print(f"Test Case: {test_case}, Expected Result: {result}")