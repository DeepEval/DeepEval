def method(text):
    words = text.split()
    ordered_words = []
    for word in words:
      ordered_word = ''.join(sorted(word))
      ordered_words.append(ordered_word)
    output = ' '.join(ordered_words)
    return output

# Test case
text = "hello world this is a test"
ordered_text = method(text)
print(f"Original text: {text}")
print(f"Ordered text: {ordered_text}")