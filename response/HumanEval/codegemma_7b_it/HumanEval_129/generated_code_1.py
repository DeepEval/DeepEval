import heapq

def method():
    # Get input from the user
    # N, k = map(int, input().split())
    N, k = 3, 5
    grid = []
    for i in range(N):
        # grid.append(list(map(int, input().split())))
        grid.append([1, 2, 3])

    # Initialize the priority queue with the first cell
    queue = [(0, 0, 0)]  # (distance, row, col)

    # Initialize the visited set
    visited = set()

    # Initialize the path
    path = []

    # Dijkstra's algorithm
    while queue:
        distance, row, col = heapq.heappop(queue)

        # Check if we have reached the end
        if distance == k:
            path.append(grid[row][col])
            return path

        # Mark the current cell as visited
        visited.add((row, col))

        # Add the current cell to the path
        path.append(grid[row][col])

        # Get the neighbor cells
        neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]

        # Explore the neighbor cells
        for neighbor_row, neighbor_col in neighbors:
            # Check if the neighbor cell is within the grid and not visited
            if 0 <= neighbor_row < N and 0 <= neighbor_col < N and (neighbor_row, neighbor_col) not in visited:
                # Add the neighbor cell to the priority queue
                heapq.heappush(queue, (distance + 1, neighbor_row, neighbor_col))

# Test case
N = 3
k = 5
grid = [[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]]

# Call the method() function
path = method()

# Print the path
print(path)