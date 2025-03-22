def method():
    # Define a list of numbers to check for divisibility by 7
    numbers = [14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105]
    
    # Initialize an empty list to store numbers divisible by 7
    divisible_by_7 = []
    
    # Iterate through the list of numbers
    for number in numbers:
        if number % 7 == 0:
            divisible_by_7.append(number)
    
    # Prepare the output
    output = f"Numbers divisible by 7: {divisible_by_7}"
    
    return output

# Call the method for validation
print(method())