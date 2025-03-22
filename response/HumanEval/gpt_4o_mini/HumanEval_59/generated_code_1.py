def method(n):
    # Initialize the variable to store the largest prime factor
    largest_prime_factor = None
    
    # Check for the smallest prime factor (2)
    while n % 2 == 0:
        largest_prime_factor = 2
        n //= 2
        
    # Check for odd factors starting from 3
    for i in range(3, int(n**0.5) + 1, 2):
        while n % i == 0:
            largest_prime_factor = i
            n //= i
            
    # If n becomes a prime number greater than 2
    if n > 2:
        largest_prime_factor = n
        
    return largest_prime_factor

# Example test case for validation
if __name__ == "__main__":
    # Test case: The largest prime factor of 13195 is 29
    n = 13195
    output = method(n)
    print(f"The largest prime factor of {n} is: {output}")