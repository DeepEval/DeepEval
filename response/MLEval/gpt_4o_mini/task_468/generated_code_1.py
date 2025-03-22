def method(num):
    # Check if the number is divisible by 7
    if num % 7 == 0:
        return f"{num} is divisible by 7."
    else:
        return f"{num} is not divisible by 7."

# Call the method for validation
number_to_check = 14  # You can change this number for testing
output = method(number_to_check)
print(output)