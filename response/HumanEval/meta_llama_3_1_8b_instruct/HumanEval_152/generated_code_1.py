def compare(scores, guesses):
    """
    This function compares the given scores and guesses for a series of matches.
    
    Args:
        scores (list): A list of scores for each match.
        guesses (list): A list of guesses for each match.
        
    Returns:
        list: A list of absolute differences between each guess and the corresponding score.
    """
    return [abs(guess - score) for score, guess in zip(scores, guesses)]

# Test cases
print(compare([1, 2, 3, 4, 5, 1], [1, 2, 3, 4, 2, -2]))  # Output: [0, 0, 0, 0, 3, 3]
print(compare([0, 5, 0, 0, 0, 4], [4, 1, 1, 0, 0, -2]))  # Output: [4, 4, 1, 0, 0, 6]