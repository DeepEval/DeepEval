import hashlib

def method(text):
    if text == "":
        return None
    else:
        return hashlib.md5(text.encode()).hexdigest()

# Test case
text = "This is a test string."
output = method(text)
print(f"MD5 hash of '{text}' is: {output}")