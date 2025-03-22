import hashlib

def method(text: str) -> str:
    if text == "":
        return None
    else:
        hash_object = hashlib.md5(text.encode())
        return hash_object.hexdigest()