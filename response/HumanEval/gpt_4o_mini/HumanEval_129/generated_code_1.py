def method(N, k, grid):
    # Directions for moving in the grid
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # To store the minimum path found
    min_path = None
    
    def is_valid(x, y):
        return 0 <= x < N and 0 <= y < N

    def dfs(x, y, path):
        nonlocal min_path
        if len(path) == k:
            if min_path is None or path < min_path:
                min_path = path[:]
            return
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny):
                path.append(grid[nx][ny])
                dfs(nx, ny, path)
                path.pop()  # backtrack

    # Start DFS from every cell in the grid
    for i in range(N):
        for j in range(N):
            dfs(i, j, [grid[i][j]])

    return min_path

# Test case
N = 3
k = 2
grid = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

output = method(N, k, grid)
print(output)  # Example output: [1, 2], as it's the lexicographically smallest path of length 2