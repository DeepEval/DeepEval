def method():
    import re
    # Define the pattern to match the number of apples and oranges
    pattern = r'[0-9]+'
    # Match the string with the pattern
    # match = re.findall(pattern, input())
    match = re.findall(pattern, "There are 5 apples and 3 oranges in the basket.")
    # Convert the matched strings to integers
    num_apples = int(match[0])
    num_oranges = int(match[1])
    # Total number of fruits in the basket
    total_fruits = num_oranges + num_apples
    # Number of mango fruits
    num_mango = total_fruits - num_oranges - num_apples
    output = num_mango
    return output