def method():
    # Helper function to compute the sum of digits of a number
    def digit_sum(n):
        return sum(int(d) for d in str(abs(n)))  # Use abs to handle negative numbers

    # The list to be sorted
    numbers = [32, 41, 23, 10, 15, 12, 43, 22]

    # Enumerate to get both the number and its original index
    sorted_numbers = sorted(enumerate(numbers), key=lambda x: (digit_sum(x[1]), x[0]))

    # Extract the sorted values from the tuples returned by sorted()
    output = [num for index, num in sorted_numbers]

    return output

# Test case
if __name__ == "__main__":
    result = method()
    print(result)  # Output should be sorted according to the sum of digits