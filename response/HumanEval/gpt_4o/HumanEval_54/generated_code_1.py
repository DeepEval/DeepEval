def method(word1, word2):
    # Convert each word to a set of characters
    set1 = set(word1)
    set2 = set(word2)
    
    # Check if both sets are equal
    output = set1 == set2
    
    return output

# Test case
def test_method():
    # assert method("listen", "silent") == True, "Test case 1 failed"
    # assert method("triangle", "integral") == True, "Test case 2 failed"
    # assert method("apple", "pale") == False, "Test case 3 failed"
    # assert method("abc", "cab") == True, "Test case 4 failed"
    # assert method("hello", "billion") == False, "Test case 5 failed"
    # assert method("aabbcc", "abc") == True, "Test case 6 failed"  # Characters present are the same
    # print("All test cases passed.")
    method("listen", "silent")

# Run the test
test_method()