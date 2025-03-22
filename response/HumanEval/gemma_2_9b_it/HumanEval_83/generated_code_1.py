def method(n):
    """
    Given a positive integer n, return the count of the numbers of n-digit positive integers that start or end with 1. 

    For example:
    method(2) == 20
    method(3) == 200
    method(4) == 2000
    """
    if n == 1:
      return 10 
    output = 9 * 10**(n-1)
    return output
    
# Test case
n = 3
output = method(n)
print(f"For n = {n}, the count of n-digit positive integers starting or ending with 1 is: {output}")