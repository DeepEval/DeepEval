class EightPuzzle:
    """
    This class is an implementation of the classic 8-puzzle game, including methods for finding the blank tile,
    making moves, getting possible moves, and solving the puzzle using a breadth-first search algorithm.
    """

    def __init__(self, initial_state):
        """
        Initialize the initial state of the Eight Puzzle Game, stores in attribute self.initial_state.
        And set the goal state of this game, stores in self.goal_state. In this case, set the size as 3*3
        :param initial_state: a 3*3 size list of Integer, stores the initial state
        """
        self.initial_state = initial_state
        self.goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    def find_blank(self, state):
        """
        Find the blank position of current state, which is the 0 element.
        :param state: a 3*3 size list of Integer, stores the current state.
        :return i, j: two Integers, represent the coordinate of the blank block.
        """
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return i, j
        raise ValueError("Blank space (0) not found in the state.")

    def move(self, state, direction):
        """
        Find the blank block, then makes the board moves forward in the given direction.
        :param state: a 3*3 size list of Integer, stores the state before moving.
        :param direction: str, only has 4 direction 'up', 'down', 'left', 'right'
        :return new_state: a 3*3 size list of Integer, stores the state after moving.
        """
        new_state = [row[:] for row in state]  # create a copy of the state
        i, j = self.find_blank(state)

        if direction == 'up':
            if i > 0:
                new_state[i][j], new_state[i - 1][j] = new_state[i - 1][j], new_state[i][j]
        elif direction == 'down':
            if i < 2:
                new_state[i][j], new_state[i + 1][j] = new_state[i + 1][j], new_state[i][j]
        elif direction == 'left':
            if j > 0:
                new_state[i][j], new_state[i][j - 1] = new_state[i][j - 1], new_state[i][j]
        elif direction == 'right':
            if j < 2:
                new_state[i][j], new_state[i][j + 1] = new_state[i][j + 1], new_state[i][j]
        else:
            raise ValueError("Invalid move direction.")
        
        return new_state

    def get_possible_moves(self, state):
        """
        According to the current state, find all the possible moving directions.
        :param state: a 3*3 size list of Integer, stores the current state.
        :return moves: a list of str, store all the possible moving directions according to the current state.
        """
        moves = []
        i, j = self.find_blank(state)

        if i > 0: moves.append('up')
        if i < 2: moves.append('down')
        if j > 0: moves.append('left')
        if j < 2: moves.append('right')

        return moves

    def solve(self):
        """
        Use BFS algorithm to find the path solution which makes the initial state to the goal state.
        :return path: list of str, the solution to the goal state.
        """
        from collections import deque

        open_list = deque([(self.initial_state, [])])
        visited = set()
        visited.add(tuple(map(tuple, self.initial_state)))  # store the state as a tuple of tuples

        while open_list:
            current_state, path = open_list.popleft()

            if current_state == self.goal_state:
                return path

            possible_moves = self.get_possible_moves(current_state)

            for move_dir in possible_moves:
                new_state = self.move(current_state, move_dir)
                new_state_tuple = tuple(map(tuple, new_state))

                if new_state_tuple not in visited:
                    visited.add(new_state_tuple)
                    open_list.append((new_state, path + [move_dir]))

        return []  # return empty if no solution found

# Test Cases
if __name__ == "__main__":
    initial_state = [[2, 3, 4], [5, 8, 1], [6, 0, 7]]
    eightPuzzle = EightPuzzle(initial_state)
    
    # Test find_blank
    blank_position = eightPuzzle.find_blank(initial_state)
    print(blank_position)  # Output: (2, 1)

    # Test move
    new_state = eightPuzzle.move(initial_state, 'left')
    print(new_state)  # Output: [[2, 3, 4], [5, 8, 1], [0, 6, 7]]

    # Test get_possible_moves
    possible_moves = eightPuzzle.get_possible_moves(initial_state)
    print(possible_moves)  # Output: ['up', 'left', 'right']

    # Test solve
    solution_path = eightPuzzle.solve()
    print(solution_path)  # Output: The moves to solve the puzzle