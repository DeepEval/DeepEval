class ChandrasekharSieve:
    """
    This is a class that uses the Chandrasekhar's Sieve method to find all prime numbers within the range.
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
        if self.n < 2:
            return []

        # Initialize a boolean array to track prime status
        is_prime = [True] * (self.n + 1)
        is_prime[0] = is_prime[1] = False  # 0 and 1 are not prime numbers

        p = 2
        while p * p <= self.n:
            if is_prime[p]:
                for i in range(p * p, self.n + 1, p):
                    is_prime[i] = False
            p += 1

        # Collecting all prime numbers
        return [num for num, prime in enumerate(is_prime) if prime]

    def get_primes(self):
        """
        Get the list of generated prime numbers.
        :return: list, a list of prime numbers
        """
        return self.primes


if __name__ == "__main__":
    # Test case for generate_primes method
    cs = ChandrasekharSieve(20)
    output_generate_primes = cs.generate_primes()
    print("Generated Primes:", output_generate_primes)  # Expected: [2, 3, 5, 7, 11, 13, 17, 19]

    # Test case for get_primes method
    output_get_primes = cs.get_primes()
    print("Get Primes:", output_get_primes)  # Expected: [2, 3, 5, 7, 11, 13, 17, 19]