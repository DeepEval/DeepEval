def method(original_string, substring):
    count = 0
    start = 0
    
    while True:
        start = original_string.find(substring, start)
        if start == -1:  # No more occurrences found
            break
        count += 1
        start += 1  # Move to the next character for overlapping cases
    
    return count

# Test case for validation
if __name__ == "__main__":
    original_string = "ABABABAB"
    substring = "AB"
    output = method(original_string, substring)
    print(f"The substring '{substring}' occurs {output} times in the original string '{original_string}'.")