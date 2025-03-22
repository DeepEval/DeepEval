def is_palindrome(num):
    return str(num) == str(num)[::-1]

def method(n):
    even_palindromes = []
    odd_palindromes = []
    
    for i in range(1, n + 1):
        if is_palindrome(i):
            if i % 2 == 0:
                even_palindromes.append(i)
            else:
                odd_palindromes.append(i)
    
    return (len(even_palindromes), len(odd_palindromes))

# Test case
output = method(10)
print(output)  # Expected output: (1, 5)