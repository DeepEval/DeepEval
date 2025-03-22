def tribonacci(n, memo={}):
    """
    Calculate the nth Tribonacci number.

    Args:
    n (int): The position of the Tribonacci number to calculate.
    memo (dict): A dictionary to store the results of previous calculations.

    Returns:
    int: The nth Tribonacci number.
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 3
    elif n == 2:
        return 4
    elif n % 2 == 0:
        return 1 + n / 2
    else:
        if n not in memo:
            memo[n] = tribonacci(n - 1, memo) + tribonacci(n - 2, memo) + tribonacci(n + 1, memo)
        return memo[n]


def method():
    """
    Generate the Tribonacci sequence up to a given position.

    Returns:
    list: The Tribonacci sequence up to the given position.
    """
    n = 10  # The position to which the sequence is generated
    sequence = []
    for i in range(1, n + 1):
        sequence.append(tribonacci(i))
    return sequence


output = method()
print(output)