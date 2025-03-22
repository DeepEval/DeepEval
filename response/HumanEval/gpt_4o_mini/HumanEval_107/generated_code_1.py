def is_palindrome(num):
    """Check if a number is a palindrome."""
    return str(num) == str(num)[::-1]

def method(n):
    even_count = 0
    odd_count = 0
    
    for i in range(1, n + 1):
        if is_palindrome(i):
            if i % 2 == 0:
                even_count += 1
            else:
                odd_count += 1
    
    return (even_count, odd_count)

# Test case
n = 100
output = method(n)
print("Even and Odd palindrome counts up to", n, ":", output)  # Expected output: (even_count, odd_count)