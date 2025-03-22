def method(n):
    count = 0
    # Iterate through each number less than n
    for i in range(n):
        # Check if the number is divisible by 11 or 13
        if i % 11 == 0 or i % 13 == 0:
            # Convert the number to string and count '7's
            count += str(i).count('7')
    return count

# Test case
n = 100
print(f"Number of times digit '7' appears in numbers less than {n} divisible by 11 or 13: {method(n)}")