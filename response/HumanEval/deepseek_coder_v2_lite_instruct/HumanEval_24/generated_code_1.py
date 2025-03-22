def method(n):
    # Start from n-1 and go down to 1 to find the largest number that divides n evenly
    for i in range(n-1, 0, -1):
        if n % i == 0:
            return i
    return 1  # If no number is found, return 1 (though this case should not occur for n > 1)

# Test case
n = 15
output = method(n)
print(f"The largest number that divides {n} evenly, smaller than {n}, is {output}")