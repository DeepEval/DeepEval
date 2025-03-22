def method(scores, guesses):
    # Check if the lengths of scores and guesses are the same
    if len(scores) != len(guesses):
        raise ValueError("Scores and guesses must be of the same length.")
    
    # Calculate the absolute differences
    output = [abs(score - guess) for score, guess in zip(scores, guesses)]
    
    return output

# Test cases for validation
print(method([1, 2, 3, 4, 5, 1], [1, 2, 3, 4, 2, -2]))  # Expected output: [0, 0, 0, 0, 3, 3]
print(method([0, 5, 0, 0, 0, 4], [4, 1, 1, 0, 0, -2]))  # Expected output: [4, 4, 1, 0, 0, 6]