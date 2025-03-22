def method():
    def is_prime(num):
        """
        Helper function to check if a number is prime.
        
        Args:
        num (int): The number to check.
        
        Returns:
        bool: True if the number is prime, False otherwise.
        """
        if num < 2:
            return False
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                return False
        return True

    def largest_prime_factor(n):
        """
        Find the largest prime factor of a number.
        
        Args:
        n (int): The number to find the largest prime factor of.
        
        Returns:
        int: The largest prime factor of the number.
        """
        i = 2
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
        return n

    output = largest_prime_factor(13195)
    return output

output = method()
print(f"The final output is: {output}")