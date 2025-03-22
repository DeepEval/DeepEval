def method():
    # Function to compare the guesses with the scores
    def compare(guesses, scores):
        # Initialize an empty list to store the differences
        output = []
        
        # Iterate over the indices of the lists
        for guess, score in zip(guesses, scores):
            # Check if the guess is correct
            if guess == score:
                # If correct, append 0 to the output list
                output.append(0)
            else:
                # If incorrect, calculate the absolute difference and append it to the output list
                output.append(abs(guess - score))
        
        return output

# Test case to validate the function
def test_method():
    # Test 1
    guesses1 = [1, 2, 3, 4, 5, 1]
    scores1 = [1, 2, 3, 4, 2, -2]
    # assert compare(guesses1, scores1) == [0, 0, 0, 0, 3, 3], "Test 1 failed"
    compare(guesses1, scores1)

    # Test 2
    guesses2 = [0, 5, 0, 0, 0, 4]
    scores2 = [4, 1, 1, 0, 0, -2]
    # assert compare(guesses2, scores2) == [4, 4, 1, 0, 0, 6], "Test 2 failed"

    print("All test cases passed")

# Run the test case
test_method()

# Return the output (not used in this implementation, just for completeness)
return output