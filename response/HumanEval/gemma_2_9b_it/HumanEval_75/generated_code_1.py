def method(a):
    def is_prime(num):
      """Helper function to check if a number is prime."""
      if num <= 1:
        return False
      for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
          return False
      return True

    if a < 100:
      prime_factors = []
      for i in range(2, int(a**0.5) + 1):
        if a % i == 0:
          if is_prime(i):
            prime_factors.append(i)
            while a % i == 0:
              a //= i
      if a > 1 and is_prime(a):
        prime_factors.append(a)
      return len(prime_factors) == 3
    else:
      return False

# Test Case
output = method(30)
print(f"Is 30 the multiplication of 3 prime numbers? {output}")