def method(lst):
    # Sort the list
    sorted_lst = sorted(lst)
    
    # Calculate the length of the list
    n = len(sorted_lst)
    
    # If the length is even, return the average of the two middle elements
    if n % 2 == 0:
        return (sorted_lst[n//2 - 1] + sorted_lst[n//2]) / 2
    else:
        # If the length is odd, return the middle element
        return sorted_lst[n//2]

# Example usage
if __name__ == "__main__":
    # Test case
    test_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    result = method(test_list)
    print("Median of the list is:", result)