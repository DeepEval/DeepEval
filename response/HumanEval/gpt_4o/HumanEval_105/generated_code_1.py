def method(arr):
    # Map of integers to their corresponding names
    num_to_name = {
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

    # Filter numbers between 1 and 9 inclusive
    filtered_numbers = [num for num in arr if 1 <= num <= 9]
    
    # Sort the filtered numbers in descending order
    sorted_numbers = sorted(filtered_numbers, reverse=True)
    
    # Map each number to its corresponding name
    output = [num_to_name[num] for num in sorted_numbers]
    
    return output

# Test case for validation
test_array = [3, 8, 2, 4, 9, 1, 5, 10, 0, 7, 6]
print(method(test_array))