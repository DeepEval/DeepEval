def count_palindromes(n):
    def is_palindrome(num):
        return str(num) == str(num)[::-1]
    
    def is_palindrome_even(num):
        return str(num) == str(num)[::-1] and len(str(num)) % 2 == 0
    
    def is_palindrome_odd(num):
        return str(num) == str(num)[::-1] and len(str(num)) % 2 != 0
    
    count_even = sum(1 for i in range(1, n+1) if is_palindrome_even(i))
    count_odd = sum(1 for i in range(1, n+1) if is_palindrome_odd(i))
    
    return count_even, count_odd

# Test case
n = 100
output = count_palindromes(n)
print(output)  # Output should be (2, 4)