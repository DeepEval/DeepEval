def method():
    def compare(scores, guesses):
        return [abs(score - guess) if score != guess else 0 for score, guess in zip(scores, guesses)]
    
    # Test case 1
    scores1 = [1, 2, 3, 4, 5, 1]
    guesses1 = [1, 2, 3, 4, 2, -2]
    output1 = compare(scores1, guesses1)
    
    # Test case 2
    scores2 = [0, 5, 0, 0, 0, 4]
    guesses2 = [4, 1, 1, 0, 0, -2]
    output2 = compare(scores2, guesses2)
    
    # Return the results for validation
    return output1, output2

# Call the method and print the results
output1, output2 = method()
print("Test Case 1 Output:", output1)
print("Test Case 2 Output:", output2)