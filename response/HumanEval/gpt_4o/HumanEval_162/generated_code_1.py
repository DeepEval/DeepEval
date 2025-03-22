import hashlib

def method(text):
    if not text:
        return None
    
    # Create an md5 hash object
    md5_hash = hashlib.md5()
    
    # Update the hash object with the bytes of the string
    md5_hash.update(text.encode('utf-8'))
    
    # Get the hexadecimal digest of the hash
    output = md5_hash.hexdigest()
    
    return output

# Test case
text = "hello world"
print("MD5 Hash:", method(text))  # Expected output: "5eb63bbbe01eeed093cb22bb8f5acdc3"

# Test case for empty string
text = ""
print("MD5 Hash:", method(text))  # Expected output: None