def find_max(words):
    if not words:
        return ""
    
    max_unique_chars = 0
    result = ""
    
    for word in words:
        unique_chars = set(word)
        if len(unique_chars) > max_unique_chars:
            max_unique_chars = len(unique_chars)
            result = word
        elif len(unique_chars) == max_unique_chars and word < result:
            result = word
    
    return result

def method():
    output = find_max(["name", "of", "string"])
    print(output)  # Expected output: "string"
    
    output = find_max(["name", "enam", "game"])
    print(output)  # Expected output: "enam"
    
    output = find_max(["aaaaaaa", "bb", "cc"])
    print(output)  # Expected output: "aaaaaaa"
    
    return output

# Test the method function
method()