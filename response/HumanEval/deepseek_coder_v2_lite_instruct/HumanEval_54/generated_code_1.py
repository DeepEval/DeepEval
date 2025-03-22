def method():
    word1 = "listen"
    word2 = "silent"
    
    # Convert words to lists of characters
    list1 = list(word1)
    list2 = list(word2)
    
    # Sort the lists of characters
    list1.sort()
    list2.sort()
    
    # Compare the sorted lists
    if list1 == list2:
        output = True
    else:
        output = False
    
    return output

# Test case
print(method())  # Expected output: True