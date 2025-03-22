def method():
    words = ["name", "of", "string"]
    output = find_max(words)
    return output

def find_max(words):
    max_unique_count = 0
    max_word = ""
    for word in words:
        unique_chars = set(word)
        if len(unique_chars) > max_unique_count:
            max_unique_count = len(unique_chars)
            max_word = word
        elif len(unique_chars) == max_unique_count and word < max_word:
            max_word = word
    return max_word

# Test Case
result = method()
print(f"Result: {result}")