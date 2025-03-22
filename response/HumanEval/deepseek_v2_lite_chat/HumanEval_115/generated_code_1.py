from collections import deque

def method():
    # Define the size of the grid
    ROWS, COLS = 5, 5

    # Define the grid with 1s representing water and 0s representing empty wells
    grid = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]

    # Define the source well (1 represents water)
    source = (0, 0)

    # Define the destination well (1 represents water)
    dest = (ROWS - 1, COLS - 1)

    # Define the capacity of each bucket
    bucket_capacity = 10

    # Initialize BFS
    queue = deque()
    visited = [[False] * COLS for _ in range(ROWS)]

    # Enqueue the source well
    queue.append(source)
    visited[source[0]][source[1]] = True

    # Define directions for BFS (up, down, left, right)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # BFS loop
    while queue:
        x, y = queue.popleft()
        if (x, y) == dest:
            return visited[dest[0]][dest[1]]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < ROWS and 0 <= ny < COLS and not visited[nx][ny] and grid[nx][ny] == 1:
                visited[nx][ny] = True
                queue.append((nx, ny))

    # If the destination is unreachable, return -1
    return -1

# Test the function
print(method())