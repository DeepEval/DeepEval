import math

def method():
    sentence = "This is a sample sentence that contains some prime words"
    words = sentence.split()
    output = []

    for word in words:
        if is_prime(len(word)):
            output.append(word)

    return ' '.join(output)


def is_prime(num):
    if num <= 1:
        return False
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True

test_case = "This is a sample sentence that contains some prime words"
print(method())