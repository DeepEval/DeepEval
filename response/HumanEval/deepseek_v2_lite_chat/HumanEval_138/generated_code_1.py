import sys

def method():
    # Read input
    # n = int(sys.stdin.readline())

    n = 10
    
    # Initialize the DP table
    dp = [[False] * (n + 1) for _ in range(4)]
    dp[0][0] = True  # Base case: 0 can be formed with 0 even numbers
    
    # Fill the DP table
    for j in range(0, n + 1):
        for i in range(1, 4):
            for k in range(0, j + 1, 2):  # Only consider adding even numbers
                if k*(i+1) <= j:
                    dp[i][j] |= dp[i-1][j-k]  # Check previous row
    
    # The answer is in the last row of the DP table
    output = dp[-1][n]
    return output if output else "No"

# Test case
print(method())  # Expected output: True or False