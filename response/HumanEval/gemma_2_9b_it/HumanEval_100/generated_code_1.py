def method(n):
    stones = []
    level = 1
    while level <= n:
        if n % 2 == 0:  # Even
            stones.append(level * 2)
        else:  # Odd
            stones.append(level * 2 + 1)
        level += 1
    return stones

# Test case
n = 5
output = method(n)
print(f"For n = {n}, the number of stones in each level is: {output}")