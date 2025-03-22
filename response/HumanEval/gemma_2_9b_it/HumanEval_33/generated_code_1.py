def method(l):
  l_prime = []
  for i in range(len(l)):
    if i % 3 != 0:
      l_prime.append(l[i])
    else:
      l_prime.append(sorted(l[i]))
  return l_prime

# Test Case
l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
output = method(l)
print(f"Input: {l}")
print(f"Output: {output}")