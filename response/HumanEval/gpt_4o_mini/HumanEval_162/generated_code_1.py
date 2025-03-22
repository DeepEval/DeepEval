import hashlib

def method(text):
    if text == "":
        return None
    # Create an md5 hash object
    md5_hash = hashlib.md5()
    # Update the hash object with the bytes of the string
    md5_hash.update(text.encode('utf-8'))
    # Return the hexadecimal representation of the hash
    return md5_hash.hexdigest()

# Test case for validation
if __name__ == "__main__":
    test_text = "hello"
    output = method(test_text)
    print(f"MD5 hash of '{test_text}': {output}")  # Expected output: "5d41402abc4b2a76b9719d911017c592"
    
    # Test with an empty string
    empty_output = method("")
    print(f"MD5 hash of empty string: {empty_output}")  # Expected output: None