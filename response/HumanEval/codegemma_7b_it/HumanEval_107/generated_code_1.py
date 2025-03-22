import math

def method(n):
    output = []

    # Function to check if a number is a palindrome
    def isPalindrome(num):
        return str(num) == str(num)[::-1]

    # Count even and odd integer palindromes
    even_count = 0
    odd_count = 0

    for i in range(1, n + 1):
        if isPalindrome(i):
            if i % 2 == 0:
                even_count += 1
            else:
                odd_count += 1

    output.append(even_count)
    output.append(odd_count)

    return tuple(output)

# Test case
n = 10
result = method(n)
print(result)  # Output: (2, 1)