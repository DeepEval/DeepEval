def method():
    # input_string = input("Enter a string: ")
    
    # 修改为固定值
    input_string = "Hello World"

    string_length = len(input_string)

    def is_prime(num):
        """Checks if a number is prime."""
        if num <= 1:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    output = is_prime(string_length)
    return output

# Test Case
print(method())  # Output: True if the length of the input string is prime, False otherwise