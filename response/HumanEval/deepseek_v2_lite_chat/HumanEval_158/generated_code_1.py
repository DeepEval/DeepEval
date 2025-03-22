def method():
    def count_unique_characters(word):
        return len(set(word))

    def find_max_unique_word(words):
        max_unique_count = 0
        max_unique_word = ""
        for word in words:
            unique_count = count_unique_characters(word)
            if unique_count > max_unique_count or (unique_count == max_unique_count and word < max_unique_word):
                max_unique_count = unique_count
                max_unique_word = word
        return max_unique_word if max_unique_word else ""

    # Test case to validate the function
    words = ["name", "of", "string", "enam", "game", "aaaaaaa", "bb", "cc"]
    output = find_max_unique_word(words)
    return output

# Call the function and print the output
result = method()
print("The word with the maximum number of unique characters is:", result)