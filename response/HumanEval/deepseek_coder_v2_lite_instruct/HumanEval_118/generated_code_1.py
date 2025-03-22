def method():
    vowels = "aeiouAEIOU"
    # word = input("Enter a word: ")
    word = "hello"
    length = len(word)

    # Iterate from right to left, starting from the second last character
    for i in range(length - 2, 0, -1):
        if word[i] not in vowels and word[i + 1] in vowels:
            return word[i + 1]

    return ""

# Test case
output = method()
print("Output:", output)