import math

class ChandrasekharSieve:
    def __init__(self, n):
        self.n = n
        self.primes = self.generate_primes()

    def generate_primes(self):
        primes = []
        for i in range(2, self.n + 1):
            if i in primes:
                continue
            for j in range(i * i, self.n + 1, i):
                if j % i == 0:
                    break
            else:
                primes.append(i)
        return primes

    def get_primes(self):
        return self.primes
    
if __name__ == "__main__":
    
    # Test case 1:
    cs = ChandrasekharSieve(20)
    output = cs.generate_primes()
    print(output)

    # Test case 2:
    cs = ChandrasekharSieve(20)
    output = cs.get_primes()
    print(output)