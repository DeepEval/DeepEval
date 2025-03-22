def next_smallest(numbers):
    unique_numbers = sorted(set(numbers))
    if len(unique_numbers) < 2:
        return None
    return unique_numbers[1]

def method():
    test_cases = [
        ([1, 2, 3, 4, 5], 2),
        ([5, 1, 4, 3, 2], 2),
        ([], None),
        ([1, 1], None),
        ([1, 2], 2),
        ([2, 2, 1, 1, 3], 2)
    ]
    
    results = []
    for numbers, expected in test_cases:
        result = next_smallest(numbers)
        results.append((result == expected, result))
    
    all_passed = all(passed for passed, _ in results)
    output = "All test cases passed." if all_passed else "Some test cases failed."
    return output

# Running the method to get the output
output = method()
print(output)