def method(n):
    def prime_factors(num):
        factors = []
        # Check for number of 2s that divide num
        while num % 2 == 0:
            factors.append(2)
            num //= 2
        # num must be odd at this point so we can skip even numbers
        for i in range(3, int(num**0.5) + 1, 2):
            while num % i == 0:
                factors.append(i)
                num //= i
        # This condition is to check if num is a prime number greater than 2
        if num > 2:
            factors.append(num)
        return factors
    
    output = prime_factors(n)
    return output

# Test case
input_number = 60  # 60 = 2^2 * 3^1 * 5^1
output = method(input_number)
print(output)  # Expected output: [2, 2, 3, 5]