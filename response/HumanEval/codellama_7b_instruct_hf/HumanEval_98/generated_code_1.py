# Import necessary packages
import re

def method():
    # Get the input string from the user
    # s = input("Enter a string: ")
    
    # 修改
    s = "HelloFromMyAwesomePROGRAM"

    # Count the uppercase vowels in even indices
    count = 0
    for i in range(0, len(s), 2):
        if s[i].isupper() and s[i] in ["A", "E", "I", "O", "U"]:
            count += 1

    # Return the count of uppercase vowels in even indices
    return count

# Test case
input_string = "HelloFromMyAwesomePROGRAM"
output = method()
print(output)