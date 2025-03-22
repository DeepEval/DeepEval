def method(sequence):
    # Calculate the common difference assuming an arithmetic sequence
    if len(sequence) < 2:
        raise ValueError("The sequence must contain at least two terms to determine a pattern.")

    common_difference = sequence[1] - sequence[0]

    # Predict the next term
    next_term = sequence[-1] + common_difference
    
    return next_term

# Example sequence
sequence = [2, 4, 6, 8]

# Call the method function for validation
output = method(sequence)
print("The next term in the sequence is:", output)