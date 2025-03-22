def method():
    # n = int(input("Enter a positive integer: "))
    n = 10
    output = []
    
    current_stone_count = n
    for i in range(n):
        output.append(current_stone_count)
        if n % 2 == 0:
            current_stone_count += 2
        else:
            current_stone_count += 2
    
    return output

# Test case
print(method())