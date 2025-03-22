def method(sentence):
    import math

    def is_prime(n):
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    words = sentence.split()
    prime_length_words = [word for word in words if is_prime(len(word))]
    output = ' '.join(prime_length_words)
    return output

# Test case
test_sentence = "This is a test of the emergency broadcast system"
print(method(test_sentence))  # Expected output should only include words whose lengths are prime numbers.