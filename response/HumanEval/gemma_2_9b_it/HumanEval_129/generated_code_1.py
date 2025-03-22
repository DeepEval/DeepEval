from collections import deque

def method(grid, k):
    n = len(grid)
    visited = set()
    queue = deque()

    # Helper function to check if a cell is valid
    def is_valid(row, col):
        return 0 <= row < n and 0 <= col < n

    # Initialize queue with starting cell (can be any)
    start_row, start_col = 0, 0
    queue.append((start_row, start_col, [grid[start_row][start_col]], 0))
    visited.add((start_row, start_col))

    while queue:
        row, col, path, length = queue.popleft()
        
        # Check if we've reached the desired path length
        if length == k:
            return path

        # Explore neighboring cells
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_row, new_col = row + dr, col + dc

            # Check if neighbor is valid and not visited
            if is_valid(new_row, new_col) and (new_row, new_col) not in visited:
                visited.add((new_row, new_col))
                new_path = path + [grid[new_row][new_col]]
                queue.append((new_row, new_col, new_path, length + 1))

    return None  # Should not reach here if a valid path exists

# Test Case
grid = [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]

k = 3
output = method(grid, k)
print(f"Minimum Path of length {k}: {output}")