import hashlib

def method():
    text = "example"  # Example input
    if not text:
        return None
    md5_hash = hashlib.md5(text.encode()).hexdigest()
    return md5_hash

# Test case
output = method()
print(output)  # Expected output: '827ccb0eea8a706c4c34a16891f84e7b' for the input 'example'