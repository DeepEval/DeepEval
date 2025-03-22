def method():
    # apples_oranges = input("Enter the number of apples and oranges separated by space: ")

    # 修改为固定值
    apples_oranges = "5 3"

    apples, oranges = map(int, apples_oranges.split())
    # total_fruits = int(input("Enter the total number of fruits in the basket: "))
    
    # 修改为固定值
    total_fruits = 10

    mangoes = total_fruits - apples - oranges
    return mangoes

# Test Case
output = method()
print(f"The number of mangoes in the basket is: {output}")