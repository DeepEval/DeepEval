def method(word1, word2):
    # Convert the words to sets of characters
    set1 = set(word1)
    set2 = set(word2)
    
    # Compare the sizes of the sets to check if the words have the same characters
    if len(set1) == len(set2):
        return True
    else:
        return False

# Example test case
# word1 = "listen"
# word2 = "silent"
# Expected output: True, because both words have the same characters

# You can uncomment the above line and use it to test the function