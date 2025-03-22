def method():
  
    input_string = "I like Python. I am learning software engineering! I wonder if this will be fun?"
    boredoms_count = 0
    sentences = input_string.split('.').split('?').split('!')
    for sentence in sentences:
        if sentence.strip().startswith("I"):
            boredoms_count += 1
    output = boredoms_count
    return output

# Test case
result = method()
print(f"Number of boredoms: {result}")