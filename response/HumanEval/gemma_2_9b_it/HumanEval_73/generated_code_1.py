def method():
    arr = [1, 2, 3, 2, 1] 
    n = len(arr)
    dp = [[0 for _ in range(n + 1)] for _ in range(n + 1)]

    for i in range(n):
        dp[i][i] = 1

    for cl in range(2, n + 1):
        for i in range(n - cl + 1):
            j = i + cl - 1
            if arr[i] == arr[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

    output = n - dp[0][n - 1]
    return output

# Test Case
result = method()
print("Minimum changes to make the array palindromic:", result)