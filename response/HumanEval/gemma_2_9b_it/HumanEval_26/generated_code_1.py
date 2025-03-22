def method():
    numbers = [1, 2, 3, 2, 1, 4, 5, 4, 6]
    unique_numbers = []
    seen = set()

    for number in numbers:
        if number not in seen:
            unique_numbers.append(number)
            seen.add(number)

    return unique_numbers

# Test case
output = method()
print(f"Input: {numbers}")
print(f"Output: {output}")