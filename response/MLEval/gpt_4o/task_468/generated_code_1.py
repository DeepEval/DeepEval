def method(number):
    # Check if the number is divisible by 7
    if number % 7 == 0:
        output = f"{number} is divisible by 7."
    else:
        output = f"{number} is not divisible by 7."
    
    return output

# Call the method for validation
print(method(14))  # Example number that is divisible by 7
print(method(15))  # Example number that is not divisible by 7