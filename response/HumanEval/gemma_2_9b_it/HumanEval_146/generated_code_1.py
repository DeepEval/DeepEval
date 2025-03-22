def method():
    numbers = [12, 345, 789, 10, 11, 23, 4567, 901, 34, 57]
    count = 0
    for num in numbers:
        if num > 10:
            first_digit = int(str(num)[0])
            last_digit = int(str(num)[-1])
            if first_digit % 2 != 0 and last_digit % 2 != 0:
                count += 1
    return count

output = method()
print(f"The number of elements meeting the criteria is: {output}")