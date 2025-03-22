def method():
    # Example input: "apples=5, oranges=3" and total_fruits = 10
    input_string = "apples=5, oranges=3"
    total_fruits = 10

    # Extract the number of apples and oranges from the input string
    apples = int(input_string.split('=')[1].split(', ')[0])
    oranges = int(input_string.split('=')[2].split(', ')[0])

    # Calculate the number of mango fruits
    mango_fruits = total_fruits - apples - oranges

    # Return the number of mango fruits
    return mango_fruits

# Test case
output = method()
print(f"The number of mango fruits in the basket is: {output}")