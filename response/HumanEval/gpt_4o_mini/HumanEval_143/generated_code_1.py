def is_prime(num):
    """Check if a number is prime."""
    if num <= 1:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def method(sentence):
    """Return a string containing words with prime lengths."""
    words = sentence.split()
    prime_length_words = [word for word in words if is_prime(len(word))]
    output = ' '.join(prime_length_words)
    return output

# Test case for validation
test_sentence = "This is a simple test of the prime lengths"
output = method(test_sentence)
print(output)  # Expected output should be words with prime lengths: "is a of the"