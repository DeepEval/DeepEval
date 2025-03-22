from typing import List

def is_prime(num: int) -> bool:
    """Check if a number is prime."""
    if num <= 1:
        return False
    if num <= 3:
        return True
    if num % 2 == 0 or num % 3 == 0:
        return False
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6
    return True

def method(n: int) -> List[int]:
    """Generate an array of the first n prime numbers less than n."""
    if n < 2:
        return []
    
    output = []
    for i in range(2, n):
        if is_prime(i):
            output.append(i)
    return output

# Test case to validate the function
def test_method():
    # assert method(10) == [2, 3, 5, 7]
    # assert method(1) == []
    # assert method(20) == [2, 3, 5, 7, 11, 13, 17, 19]
    print(method(1))

test_method()