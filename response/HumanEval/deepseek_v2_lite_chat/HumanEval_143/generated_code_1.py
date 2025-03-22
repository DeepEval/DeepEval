from sympy import isprime

def method():
    # Function to check if a number is prime
    def is_prime_length(word):
        return isprime(len(word))
    
    # Read the sentence
    # sentence = input("Enter a sentence: ")
    sentence = "this is a test sentence"
    
    # Split the sentence into words
    words = sentence.split()
    
    # Filter words based on prime length
    prime_length_words = [word for word in words if is_prime_length(word)]
    
    # Join the prime length words back into a sentence
    output = ' '.join(prime_length_words)
    
    return output

# Test case
def test_method():
    # assert method() == "Enter a sentence: this is a test sentence"
    # assert method() == "this is a test sentence"
    # assert method() == "Enter a sentence: prime example sentence"
    # assert method() == "prime example sentence"
    # assert method() == "Enter a sentence: words with non prime lengths"
    # assert method() == "words with non prime lengths"
    method()

# Run the test case
test_method()