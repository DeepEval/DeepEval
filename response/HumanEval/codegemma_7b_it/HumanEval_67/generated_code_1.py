import re

def method():

    # fruit_str = input("Enter the number of oranges and apples in the basket: ")
    # total_fruits = int(input("Enter the total number of fruits in the basket: "))
    fruit_str = "10 apples and 5 oranges"
    total_fruits = 15
    
    fruit_pattern = re.compile(r"(\d+) apples? and (\d+) oranges?")
    match = fruit_pattern.match(fruit_str)

    if match:
        apples = int(match.group(1))
        oranges = int(match.group(2))
        mangoes = total_fruits - apples - oranges
        return mangoes
    else:
        return "Invalid input format."


# Test case
fruit_str = "10 apples and 5 oranges"
total_fruits = 15

result = method()

print(result)