def is_prime(n):
    """Check if a number is prime."""
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

def method():
    numbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    largest_prime = -1
    
    for num in numbers:
        if is_prime(num) and num > largest_prime:
            largest_prime = num
    
    if largest_prime == -1:
        return 0  # No prime numbers found
    
    sum_of_digits = sum(int(digit) for digit in str(largest_prime))
    return sum_of_digits

# Test case
output = method()
print(output)  # Expected output: 10 (since the largest prime number is 97 and the sum of its digits is 16)