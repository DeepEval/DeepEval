def method(n):
    output = []
    for i in range(1, n + 1):
        if i % 2 == 0:  
            fact = 1
            for j in range(1, i + 1):
                fact *= j
            output.append(fact)
        else:
            sum = 0
            for j in range(1, i + 1):
                sum += j
            output.append(sum)
    return output

# Test case
n = 5
result = method(n)
print(f"For n = {n}, the output is: {result}")