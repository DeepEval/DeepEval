import math

def method():
    # Function to check if a number is the multiplication of 3 prime numbers
    def is_prime(num):
        # Function to check if a number is prime
        if num <= 1:
            return False
        for i in range(2, int(math.sqrt(num)) + 1):
            if num % i == 0:
                return False
        return True

    # Function to check if a number is the multiplication of 3 prime numbers
    def is_multiple_of_3_primes(num):
        # Check if the number is less than 100
        if num >= 100:
            return False
        # Check if the number can be divided by 2
        if num % 2 == 0:
            return False
        # Check if the number can be divided by 3
        if num % 3 == 0:
            return False
        # Check if the number can be divided by 5
        if num % 5 == 0:
            return False
        # Check if the number can be divided by 7
        if num % 7 == 0:
            return False
        # Check if the number can be divided by 11
        if num % 11 == 0:
            return False
        # Check if the number can be divided by 13
        if num % 13 == 0:
            return False
        # Check if the number can be divided by 17
        if num % 17 == 0:
            return False
        # Check if the number can be divided by 19
        if num % 19 == 0:
            return False
        # Check if the number can be divided by 23
        if num % 23 == 0:
            return False
        # Check if the number can be divided by 29
        if num % 29 == 0:
            return False
        # Check if the number can be divided by 31
        if num % 31 == 0:
            return False
        # Check if the number can be divided by 37
        if num % 37 == 0:
            return False
        # Check if the number can be divided by 41
        if num % 41 == 0:
            return False
        # Check if the number can be divided by 43
        if num % 43 == 0:
            return False
        # Check if the number can be divided by 47
        if num % 47 == 0:
            return False
        # Check if the number can be divided by 53
        if num % 53 == 0:
            return False
        # Check if the number can be divided by 59
        if num % 59 == 0:
            return False
        # Check if the number can be divided by 61
        if num % 61 == 0:
            return False
        # Check if the number can be divided by 67
        if num % 67 == 0:
            return False
        # Check if the number can be divided by 71
        if num % 71 == 0:
            return False
        # Check if the number can be divided by 73
        if num % 73 == 0:
            return False
        # Check if the number can be divided by 79
        if num % 79 == 0:
            return False
        # Check if the number can be divided by 83
        if num % 83 == 0:
            return False
        # Check if the number can be divided by 89
        if num % 89 == 0:
            return False
        # Check if the number can be divided by 97
        if num % 97 == 0:
            return False
        # The number is not the multiplication of 3 prime numbers
        return False

    # Test case
    test_case = 3*5*7
    result = is_multiple_of_3_primes(test_case)
    print(result)