from collections import deque

def method():
    def bfs(grid, start, end, k):
        n = len(grid)
        queue = deque([(start, [start], 0)])
        visited = {start}
        while queue:
            cell, path, path_length = queue.popleft()
            if cell == end:
                return path + [end]
            for neighbor in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                new_cell = (cell[0] + neighbor[0], cell[1] + neighbor[1])
                if 0 <= new_cell[0] < n and 0 <= new_cell[1] < n and new_cell not in visited:
                    visited.add(new_cell)
                    new_path = path + [new_cell]
                    queue.append((new_cell, new_path, path_length + 1))
        return []

    def lexicographical_compare(lst_a, lst_b):
        for a, b in zip(lst_a, lst_b):
            if a > b:
                return False
        return True

    def find_min_path(grid, k):
        n = len(grid)
        for start in [(i, j) for i in range(n) for j in range(n) if all(grid[i][j] == val)]:
            if all(grid[start[0]][start[1]] == val for val in range(1, n * n + 1)):
                end = (start[0] + 1, start[1]) if start[0] < n - 1 else (start[0], start[1] + 1)
                path = bfs(grid, start, end, k)
                if lexicographical_compare(path, path[:k]):
                    return path
        return []

    # Read input from stdin
    # n, k = map(int, input().split())
    # grid = [list(map(int, input().split())) for _ in range(n)]
    n, k = 3, 5
    grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    # Find the minimum path
    output = find_min_path(grid, k)
    return output

# Test case
test_case = """
3 5
1 2 3
4 5 6
7 8 9
"""

# Run the method with the test case
output = method()

# Print the output
print(output)