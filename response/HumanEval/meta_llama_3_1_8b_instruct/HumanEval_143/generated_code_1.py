import sympy  # Import the sympy library to check for prime numbers

def method(sentence):
    """
    This function takes a sentence as input, filters out the words with prime lengths, 
    and returns a new string with the remaining words in the same order.

    Args:
        sentence (str): The input sentence.

    Returns:
        str: The new string with words of prime lengths.
    """

    # Split the sentence into words
    words = sentence.split()

    # Filter out the words with prime lengths using sympy.isprime function
    prime_words = [word for word in words if sympy.isprime(len(word))]

    # Join the prime words back into a string separated by spaces
    output =''.join(prime_words)

    return output


# Test case
sentence = "I am a developer specializing in software engineering"
print(method(sentence))