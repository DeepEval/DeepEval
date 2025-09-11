from collections import deque

class EightPuzzle:
    """
    This class is an implementation of the classic 8-puzzle game, including methods for finding the blank tile,
    making moves, getting possible moves, and solving the puzzle using a breadth-first search algorithm.
    """

    def __init__(self, initial_state):
        """
        Initializing the initial state of the Eight Puzzle Game.
        :param initial_state: a 3*3 size list of Integer, stores the initial state
        """
        self.initial_state = initial_state
        self.goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    def find_blank(self, state):
        """
        Find the blank position of the current state, which is the 0 element.
        :param state: a 3*3 size list of Integer, stores the current state.
        :return: i, j: two Integers, represent the coordinate of the blank block.
        """
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return i, j

    def move(self, state, direction):
        """
        Find the blank block, then make the board move towards the given direction.
        :param state: a 3*3 size list of Integer, stores the state before moving.
        :param direction: str, only has 4 directions 'up', 'down', 'left', 'right'
        :return: new_state: a 3*3 size list of Integer, stores the state after moving.
        """
        i, j = self.find_blank(state)
        new_state = [row[:] for row in state]  # Make a deep copy of the state

        if direction == 'up' and i > 0:
            new_state[i][j], new_state[i-1][j] = new_state[i-1][j], new_state[i][j]
        elif direction == 'down' and i < 2:
            new_state[i][j], new_state[i+1][j] = new_state[i+1][j], new_state[i][j]
        elif direction == 'left' and j > 0:
            new_state[i][j], new_state[i][j-1] = new_state[i][j-1], new_state[i][j]
        elif direction == 'right' and j < 2:
            new_state[i][j], new_state[i][j+1] = new_state[i][j+1], new_state[i][j]

        return new_state

    def get_possible_moves(self, state):
        """
        According to the current state, find all the possible moving directions.
        :param state: a 3*3 size list of Integer, stores the current state.
        :return: moves: a list of str, store all the possible moving directions according to the current state.
        """
        possible_moves = []
        i, j = self.find_blank(state)

        if i > 0:
            possible_moves.append('up')
        if i < 2:
            possible_moves.append('down')
        if j > 0:
            possible_moves.append('left')
        if j < 2:
            possible_moves.append('right')

        return possible_moves

    def solve(self):
        """
        Use BFS algorithm to find the path solution which makes the initial state to the goal state.
        :return: path: list of str, the solution to the goal state.
        """
        queue = deque([(self.initial_state, [])])
        visited = set()

        while queue:
            current_state, path = queue.popleft()
            if current_state == self.goal_state:
                return path

            state_tuple = tuple(tuple(row) for row in current_state)
            if state_tuple in visited:
                continue

            visited.add(state_tuple)

            for move in self.get_possible_moves(current_state):
                new_state = self.move(current_state, move)
                queue.append((new_state, path + [move]))

        return []

if __name__ == "__main__":
    # Test find_blank
    eightPuzzle = EightPuzzle([[2, 3, 4], [5, 8, 1], [6, 0, 7]])
    output = eightPuzzle.find_blank([[2, 3, 4], [5, 8, 1], [6, 0, 7]])
    print("find_blank:", output)  # Expected: (2, 1)

    # Test move
    output = eightPuzzle.move([[2, 3, 4], [5, 8, 1], [6, 0, 7]], 'left')
    print("move:", output)  # Expected: [[2, 3, 4], [5, 8, 1], [0, 6, 7]]

    # Test get_possible_moves
    output = eightPuzzle.get_possible_moves([[2, 3, 4], [5, 8, 1], [6, 0, 7]])
    print("get_possible_moves:", output)  # Expected: ['up', 'left', 'right']

    # Test solve
    eightPuzzle = EightPuzzle([[1, 2, 3], [4, 5, 6], [7, 0, 8]])
    output = eightPuzzle.solve()
    print("solve:", output)  # Expected: ['right']