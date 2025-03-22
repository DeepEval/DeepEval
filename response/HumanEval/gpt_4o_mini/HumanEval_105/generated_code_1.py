def method(arr):
    # Step 1: Filter integers between 1 and 9
    filtered = [num for num in arr if 1 <= num <= 9]
    
    # Step 2: Sort the filtered list
    sorted_filtered = sorted(filtered)
    
    # Step 3: Reverse the sorted list
    reversed_sorted = sorted_filtered[::-1]
    
    # Mapping from numbers to their corresponding names
    number_to_name = {
        1: "One", 2: "Two", 3: "Three", 4: "Four", 
        5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 
        9: "Nine"
    }
    
    # Step 4: Replace each digit by its corresponding name
    output = [number_to_name[num] for num in reversed_sorted]
    
    return output

# Test case
test_array = [5, 3, 8, 2, 10, 1, 6, -1, 0, 4, 7, 11, 9]
output = method(test_array)
print(output)  # Expected output: ['Nine', 'Eight', 'Seven', 'Six', 'Five', 'Four', 'Three', 'Two', 'One']