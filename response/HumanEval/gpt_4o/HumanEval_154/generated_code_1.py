def cycpattern_check(word1, word2):
    # Generate all rotations of word2
    rotations = [word2[i:] + word2[:i] for i in range(len(word2))]
    
    # Check if any rotation is a substring of word1
    for rotation in rotations:
        if rotation in word1:
            return True
    return False

def method():
    # Test cases
    assert cycpattern_check("abcd", "abd") == False
    assert cycpattern_check("hello", "ell") == True
    assert cycpattern_check("whassup", "psus") == False
    assert cycpattern_check("abab", "baa") == True
    assert cycpattern_check("efef", "eeff") == False
    assert cycpattern_check("himenss", "simen") == True
    
    output = "All test cases passed"
    return output

# Running the method to validate
print(method())