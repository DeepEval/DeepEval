def method(n):
    """
    Given a positive integer n, return a sorted list that has the odd numbers in collatz sequence. 

    The Collatz conjecture is a conjecture in mathematics that concerns a sequence defined as follows:
    start with any positive integer n. Then each term is obtained from the previous term as follows:
    if the previous term is even, the next term is one half of the previous term.
    If the previous term is odd, the next term is 3 times the previous term plus 1.

    The conjecture is that no matter what value of n, the sequence will always reach 1.

    Note: 
    1. Collatz(1) is [1].
    2. returned list sorted in increasing order.
    """

    collatz_sequence = []
    while n != 1:
        if n % 2 != 0:  # If n is odd
            collatz_sequence.append(n)
            n = 3 * n + 1  
        else:  # If n is even
            n = n // 2
    
    collatz_sequence.append(1) # Append 1 to the sequence
    return sorted(collatz_sequence)

# Test Case
n = 10
output = method(n)
print(f"Collatz sequence for {n}: {output}")