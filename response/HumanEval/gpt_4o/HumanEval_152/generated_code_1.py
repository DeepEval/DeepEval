def compare(scores, guesses):
    # Initialize an empty list to store the differences
    differences = []
    
    # Iterate over the scores and guesses simultaneously
    for score, guess in zip(scores, guesses):
        # Calculate the absolute difference and add it to the list
        differences.append(abs(score - guess))
    
    return differences

# Example test cases
print(compare([1, 2, 3, 4, 5, 1], [1, 2, 3, 4, 2, -2]))  # Output: [0, 0, 0, 0, 3, 3]
print(compare([0, 5, 0, 0, 0, 4], [4, 1, 1, 0, 0, -2]))  # Output: [4, 4, 1, 0, 0, 6]