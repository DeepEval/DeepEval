class ChandrasekharSieve:
    def __init__(self, n):
        self.n = n
        self.primes = self.generate_primes()

    def generate_primes(self):
        sieve = [True] * (self.n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(self.n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, self.n + 1, i):
                    sieve[j] = False
        return [i for i in range(2, self.n) if sieve[i]]

    def get_primes(self):
        return self.primes

# Example usage:
if __name__ == "__main__":
    cs = ChandrasekharSieve(20)
    print(cs.get_primes())  # Output: [2, 3, 5, 7, 11, 13, 17, 19]