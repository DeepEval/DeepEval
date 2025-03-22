def method():
    """
    Calculates the Tribonacci sequence up to a given number of terms.

    Returns:
        list: A list containing the Tribonacci sequence up to the specified number of terms.
    """

    n_terms = 10  # You can change this to calculate more or fewer terms
    tribonacci_sequence = [3]  # Initialize the sequence with the first term

    for i in range(2, n_terms):
        if i % 2 == 0:
            tribonacci_sequence.append(1 + i // 2)
        else:
            tribonacci_sequence.append(tribonacci_sequence[i-1] + tribonacci_sequence[i-2] + tribonacci_sequence[i])

    return tribonacci_sequence

# Test Case
output = method()
print(f"Tribonacci sequence: {output}")