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
        >>> cs = ChandrasekharSieve(20)
        >>> cs.generate_primes()
        [2, 3, 5, 7, 11, 13, 17, 19]

        """
        prime = [True] * (self.n + 1)
        result = []
        p = 2
        while (p * p <= self.n):
            if (prime[p] == True):
                for i in range(p * p, self.n + 1, p):
                    prime[i] = False
            p += 1
        for p in range(2, self.n + 1):
            if prime[p]:
                result.append(p)
        return result

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
    instance = ChandrasekharSieve(20)
    # Test case for __init__
    print(instance.get_primes())  

    # Test case for generate_primes
    print(instance.generate_primes())  

    # Test case for get_primes
    print(instance.get_primes())