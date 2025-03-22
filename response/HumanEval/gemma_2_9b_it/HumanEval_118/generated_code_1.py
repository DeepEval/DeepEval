def method():
    # word = input("Enter a word: ")
    # 修改为固定值
    word = "consonant"
    
    vowels = "aeiouAEIOU"
    output = ""
    for i in range(len(word) - 2, -1, -1):
        if word[i] in vowels and word[i + 1] not in vowels and word[i + 2] not in vowels:
            output = word[i]
            break
    return output

# Test case
test_word = "consonant"
closest_vowel = method()
print(f"Closest vowel in '{test_word}' is: {closest_vowel}")