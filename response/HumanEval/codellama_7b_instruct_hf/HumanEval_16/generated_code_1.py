import re

def method(str):
    str = re.sub(r'[^\w\s]', '', str).lower()
    return len(set(str))

test_str = "This is A string with punctuation!"
print(method(test_str))