def method():
    # n = int(input("Enter the number of cars in each direction: ")) 
    
    # 修改为固定值
    n = 5

    car_left = list(range(1, n + 1))  # Cars moving left to right
    car_right = list(range(n, 2 * n))  # Cars moving right to left

    collisions = 0
    i = 0
    j = 0
    while i < len(car_left) and j < len(car_right):
        if car_left[i] < car_right[j]:
            i += 1
        elif car_left[i] > car_right[j]:
            j += 1
        else:
            collisions += 1
            i += 1
            j += 1

    output = collisions
    return output

# Test Case
result = method()
print("Number of collisions:", result)