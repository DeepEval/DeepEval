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
        if self.n < 2:
            return []

        # Initialize a list to mark prime status of numbers
        is_prime = [True] * (self.n + 1)
        is_prime[0] = is_prime[1] = False  # 0 and 1 are not prime numbers

        p = 2
        while p * p <= self.n:
            if is_prime[p]:
                for multiple in range(p * p, self.n + 1, p):
                    is_prime[multiple] = False
            p += 1

        # Extract primes from the is_prime list
        primes = [p for p in range(self.n + 1) if is_prime[p]]
        return primes

    def get_primes(self):
        """
        Get the list of generated prime numbers.
        :return: list, a list of prime numbers
        """
        return self.primes

# Test cases
if __name__ == "__main__":
    cs = ChandrasekharSieve(20)
    print(cs.get_primes())  # Expected output: [2, 3, 5, 7, 11, 13, 17, 19]

    cs = ChandrasekharSieve(30)
    print(cs.get_primes())  # Expected output: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]