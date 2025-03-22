def method(n):
    def is_palindrome(num):
        s = str(num)
        return s == s[::-1]
    
    even_count = 0
    odd_count = 0
    
    for i in range(1, n + 1):
        if is_palindrome(i):
            if i % 2 == 0:
                even_count += 1
            else:
                odd_count += 1
    
    output = (even_count, odd_count)
    return output

# Test case
n = 100
print(method(n))  # Example test case