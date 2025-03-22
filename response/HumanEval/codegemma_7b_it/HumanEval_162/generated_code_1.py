import hashlib

def method():
    text = "This is a text to be hashed"
    if not text:
        return None

    md5_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    return md5_hash


test_case = method()
print(test_case)