from collections import deque

class EightPuzzle:
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    def find_blank(self, state):
        for i, row in enumerate(state):
            for j, value in enumerate(row):
                if value == 0:
                    return i, j

    def move(self, state, direction):
        blank = self.find_blank(state)
        i, j = blank
        if direction == 'up':
            state[i][j], state[i-1][j] = state[i-1][j], state[i][j]
        elif direction == 'down':
            state[i][j], state[i+1][j] = state[i+1][j], state[i][j]
        elif direction == 'left':
            state[i][j], state[i][j-1] = state[i][j-1], state[i][j]
        elif direction == 'right':
            state[i][j], state[i][j+1] = state[i][j+1], state[i][j]
        return state

    def get_possible_moves(self, state):
        moves = ['up', 'down', 'left', 'right']
        blank = self.find_blank(state)
        possible_moves = []
        for direction in moves:
            if direction == 'up' and blank[0] > 0:
                possible_moves.append(direction)
            if direction == 'down' and blank[0] < 2:
                possible_moves.append(direction)
            if direction == 'left' and blank[1] > 0:
                possible_moves.append(direction)
            if direction == 'right' and blank[1] < 2:
                possible_moves.append(direction)
        return possible_moves

    def solve(self):
        open_list = deque([self.initial_state])
        closed_list = []
        came_from = {}
        cost = {}

        cost[self.initial_state] = 0
        came_from[self.initial_state] = None
        f = lambda x: cost[x] + sum(abs(i - j) for i, j in zip(x, self.goal_state))

        while open_list:
            current = open_list.popleft()
            closed_list.append(current)
            if current == self.goal_state:
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for move in self.get_possible_moves(current):
                next_node = self.move(current, move)
                if next_node not in closed_list and next_node not in open_list:
                    open_list.append(next_node)
                    came_from[next_node] = current
                    cost[next_node] = cost[current] + 1

        return []

# Test cases
if __name__ == "__main__":
    # Test find_blank
    eightPuzzle = EightPuzzle([[2, 3, 4], [5, 8, 1], [6, 0, 7]])
    print(eightPuzzle.find_blank(eightPuzzle.initial_state))  # Expected: (2, 1)

    # Test move
    eightPuzzle = EightPuzzle([[2, 3, 4], [5, 8, 1], [6, 0, 7]])
    print(eightPuzzle.move(eightPuzzle.initial_state, 'left'))  # Expected: [[2, 3, 4], [5, 8, 1], [0, 6, 7]]

    # Test get_possible_moves
    eightPuzzle = EightPuzzle([[2, 3, 4], [5, 8, 1], [6, 0, 7]])
    print(eightPuzzle.get_possible_moves(eightPuzzle.initial_state))  # Expected: ['up', 'left', 'right']

    # Test solve
    eightPuzzle = EightPuzzle([[1, 2, 3], [4, 5, 6], [7, 0, 8]])
    print(eightPuzzle.solve())  # Expected: ['right']