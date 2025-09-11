class ChandrasekharSieve:
    """
    This is a class that uses the Sieve of Eratosthenes to find all prime numbers within the range
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
        Generate prime numbers up to the specified limit using the Sieve of Eratosthenes.
        :return: list, a list of prime numbers
        >>> cs = ChandrasekharSieve(20)
        >>> cs.generate_primes()
        [2, 3, 5, 7, 11, 13, 17, 19]
        """
        if self.n < 2:
            return []

        is_prime = [True] * (self.n + 1)
        is_prime[0], is_prime[1] = False, False  # 0 and 1 are not prime numbers

        for i in range(2, int(self.n**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, self.n + 1, i):
                    is_prime[j] = False

        return [i for i in range(2, self.n + 1) if is_prime[i]]

    def get_primes(self):
        """
        Get the list of generated prime numbers.
        :return: list, a list of prime numbers
        >>> cs = ChandrasekharSieve(20)
        >>> cs.get_primes()
        [2, 3, 5, 7, 11, 13, 17, 19]
        """
        return self.primes


if __name__ == "__main__":
    # Test case for generate_primes method
    cs = ChandrasekharSieve(20)
    output_primes = cs.generate_primes()
    print("Generated Primes:", output_primes)  # Expected: [2, 3, 5, 7, 11, 13, 17, 19]

    # Test case for get_primes method
    output_get_primes = cs.get_primes()
    print("Get Primes:", output_get_primes)  # Expected: [2, 3, 5, 7, 11, 13, 17, 19]