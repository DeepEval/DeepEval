def method():
    sentence = "This is a sentence with some words"
    words = sentence.split()
    prime_words = []
    for word in words:
        if is_prime(len(word)):
            prime_words.append(word)
    output = " ".join(prime_words)
    return output

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# Test Case
output = method()
print(f"Input: {sentence}")
print(f"Output: {output}")