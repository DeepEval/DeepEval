def find_max(words):
    # Initialize variables to track the word with max unique characters
    max_unique_count = -1
    max_word = ""
    
    for word in words:
        # Calculate the number of unique characters in the current word
        unique_chars = set(word)
        unique_count = len(unique_chars)
        
        # Check if the current word has more unique characters or comes first lexicographically
        if unique_count > max_unique_count or (unique_count == max_unique_count and word < max_word):
            max_unique_count = unique_count
            max_word = word
            
    return max_word

# Test case for validation
print(find_max(["name", "of", "string"]))  # Output: "string"
print(find_max(["name", "enam", "game"]))  # Output: "enam"
print(find_max(["aaaaaaa", "bb", "cc"]))   # Output: "aaaaaaa"