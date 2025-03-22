import numpy as np

def method(scores: list, guesses: list) -> list:
    """
    Compares the scores and guesses of a number of matches and returns an array of how far off each guess was.

    Args:
        scores: A list of scores for each match.
        guesses: A list of guesses for each match.

    Returns:
        A list of the same length as scores and guesses, where each value is the absolute difference between the guess and the score.
    """

    # Convert the lists to NumPy arrays for efficient calculations
    scores = np.array(scores)
    guesses = np.array(guesses)

    # Calculate the absolute difference between the scores and guesses
    output = np.abs(guesses - scores)

    # Return the list of absolute differences
    return output


# Test case
scores = [1, 2, 3, 4, 5, 1]
guesses = [1, 2, 3, 4, 2, -2]

output = method(scores, guesses)

print(output)