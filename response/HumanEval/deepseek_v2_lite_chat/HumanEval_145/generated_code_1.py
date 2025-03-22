def method():
    def sum_of_digits(n):
        return sum(int(digit) for digit in str(n))
    
    def sort_by_sum_and_index(lst):
        return sorted(lst, key=lambda x: (sum_of_digits(x), lst.index(x)))
    
    # Example list of integers
    numbers = [42, 295, 9, 25, 14, 794, 566, 12, 38]
    
    # Sort the list based on the sum of digits and original index
    sorted_numbers = sort_by_sum_and_index(numbers)
    
    return sorted_numbers

# Example test case
if __name__ == "__main__":
    sorted_numbers = method()
    print(sorted_numbers)