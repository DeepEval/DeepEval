def method():
    numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6]
    filtered_numbers = [num for num in numbers if 1 <= num <= 9]
    sorted_numbers = sorted(filtered_numbers)
    reversed_numbers = sorted_numbers[::-1]
    
    number_names = {
        1: "One",
        2: "Two",
        3: "Three",
        4: "Four",
        5: "Five",
        6: "Six",
        7: "Seven",
        8: "Eight",
        9: "Nine"
    }
    
    output = [number_names[num] for num in reversed_numbers]
    
    return output

# Test case
print(method())