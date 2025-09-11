import numpy as np

class ChandrasekharSieve:
    """
    This is a class that uses the Chandrasekhar's Sieve method to find all prime numbers within the range
    """

    def __init__(self, n):
        """
        Initialize the ChandrasekharSieve class with the given limit.
        :param n: int, the upper limit for generating prime numbers
        """
        self.n = n
        self.primes = self.generate_primes()

    def generate_primes(self):
        """
        Generate prime numbers up to the specified limit using the Chandrasekhar sieve algorithm.
        :return: list, a list of prime numbers
        """
        # Create a boolean array, prime, of size n+1
        prime = np.ones(self.n + 1, dtype=bool)

        # 0 and 1 are not prime numbers
        prime[0] = prime[1] = False

        # Iterate from 2 to sqrt(n)
        for p in range(2, int(np.sqrt(self.n)) + 1):
            # If p is a prime number, mark its multiples as non-prime
            if prime[p]:
                for i in range(p * p, self.n + 1, p):
                    prime[i] = False

        # Return a list of prime numbers
        return [p for p in range(2, self.n + 1) if prime[p]]

    def get_primes(self):
        """
        Get the list of generated prime numbers.
        :return: list, a list of prime numbers
        """
        return self.primes

if __name__ == "__main__":
    # Test case for generate_primes method
    instance = ChandrasekharSieve(20)
    output = instance.generate_primes()
    print("Generated Primes:", output)

    # Test case for get_primes method
    instance = ChandrasekharSieve(20)
    output = instance.get_primes()
    print("List of Generated Primes:", output)