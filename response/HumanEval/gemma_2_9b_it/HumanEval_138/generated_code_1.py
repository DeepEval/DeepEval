def method(n):
    """
    Evaluates whether the given number n can be written as the sum of exactly 4 positive even numbers.

    Args:
      n: An integer.

    Returns:
      True if n can be written as the sum of 4 positive even numbers, False otherwise.
    """
    
    for i in range(2, n//2 + 1, 2):  # Iterate through possible even numbers up to n/2
        for j in range(i, (n-i)//2 + 1, 2):  # Iterate through possible even numbers >= i
            for k in range(j, (n-i-j)//2 + 1, 2):  # Iterate through possible even numbers >= j
                l = n - i - j - k  # Calculate the remaining even number
                if l > 0 and l % 2 == 0:  # Check if the remaining number is positive and even
                    return True
    return False

# Test case
n = 30
output = method(n)
print(f"Can {n} be written as the sum of 4 positive even numbers? {output}")