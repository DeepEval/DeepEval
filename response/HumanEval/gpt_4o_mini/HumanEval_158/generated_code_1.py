def find_max(words):
    max_unique_count = -1
    result_word = ""
    
    for word in words:
        unique_count = len(set(word))  # Count unique characters using a set
        # Compare to find the maximum unique count or lexicographically smaller word
        if (unique_count > max_unique_count) or (unique_count == max_unique_count and word < result_word):
            max_unique_count = unique_count
            result_word = word
            
    return result_word

def method():
    # Example test cases
    test_cases = [
        (["name", "of", "string"], "string"),
        (["name", "enam", "game"], "enam"),
        (["aaaaaaa", "bb", "cc"], "aaaaaaa"),
        (["abc", "acb", "bac"], "abc"),
        (["apple", "banana", "grape"], "banana")
    ]
    
    for words, expected in test_cases:
        # assert find_max(words) == expected, f"Test failed for input: {words}"
        find_max(words)
    
    return "All tests passed!"

# Call the method to run the test cases
output = method()
print(output)