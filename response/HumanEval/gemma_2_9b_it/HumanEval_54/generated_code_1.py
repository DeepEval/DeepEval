def method():
    # word1 = input("Enter the first word: ")
    # word2 = input("Enter the second word: ")

    # 修改为固定值
    word1 = "listen"
    word2 = "silent"

    word1_sorted = sorted(word1)
    word2_sorted = sorted(word2)

    if word1_sorted == word2_sorted:
        output = f"Both words '{word1}' and '{word2}' have the same characters."
    else:
        output = f"The words '{word1}' and '{word2}' do not have the same characters."
    
    return output

# Test case
print(method())