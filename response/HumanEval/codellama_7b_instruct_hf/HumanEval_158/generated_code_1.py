from collections import Counter

def find_max(str_list):
    max_unique_chars = 0
    max_word = ""

    for word in str_list:
        unique_chars = len(Counter(word))
        if unique_chars > max_unique_chars:
            max_unique_chars = unique_chars
            max_word = word
        elif unique_chars == max_unique_chars:
            max_word = min(max_word, word)

    return max_word

# Test case
str_list = ["name", "of", "string"]
max_word = find_max(str_list)
print(max_word)  # Output: string