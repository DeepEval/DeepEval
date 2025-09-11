from collections import deque

class EightPuzzle:
    """
    This class is an implementation of the classic 8-puzzle game, including methods for finding the blank tile, making moves, getting possible moves, and solving the puzzle using a breadth-first search algorithm.
    """

    def __init__(self, initial_state):
        """
        Initializing the initial state of Eight Puzzle Game, stores in attribute self.initial_state.
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
        >>> eightPuzzle = EightPuzzle([[2, 3, 4], [5, 8, 1], [6, 0, 7]])
        >>> eightPuzzle.find_blank([[2, 3, 4], [5, 8, 1], [6, 0, 7]])
        (2, 1)
        """
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return (i, j)

    def move(self, state, direction):
        """
        Find the blank block, then makes the board moves forward the given direction.
        :param state: a 3*3 size list of Integer, stores the state before moving.
        :param direction: str, only has 4 direction 'up', 'down', 'left', 'right'
        :return new_state: a 3*3 size list of Integer, stores the state after moving.
        >>> eightPuzzle = EightPuzzle([[2, 3, 4], [5, 8, 1], [6, 0, 7]])
        >>> eightPuzzle.move([[2, 3, 4], [5, 8, 1], [6, 0, 7]], 'left')
        [[2, 3, 4], [5, 8, 1], [0, 6, 7]]
        """
        i, j = self.find_blank(state)
        if direction == 'up' and i > 0:
            state[i][j], state[i - 1][j] = state[i - 1][j], state[i][j]
        elif direction == 'down' and i < 2:
            state[i][j], state[i + 1][j] = state[i + 1][j], state[i][j]
        elif direction == 'left' and j > 0:
            state[i][j], state[i][j - 1] = state[i][j - 1], state[i][j]
        elif direction == 'right' and j < 2:
            state[i][j], state[i][j + 1] = state[i][j + 1], state[i][j]
        return state

    def get_possible_moves(self, state):
        """
        According the current state, find all the possible moving directions. Only has 4 direction 'up', 'down', 'left', 'right'.
        :param state: a 3*3 size list of Integer, stores the current state.
        :return moves: a list of str, store all the possible moving directions according to the current state.
        >>> eightPuzzle.get_possible_moves([[2, 3, 4], [5, 8, 1], [6, 0, 7]])
        ['up', 'left', 'right']
        """
        i, j = self.find_blank(state)
        moves = []
        if i > 0:
            moves.append('up')
        if j < 2:
            moves.append('right')
        if i < 2:
            moves.append('down')
        if j > 0:
            moves.append('left')
        return moves

    def solve(self):
        """
        Use BFS algorithm to find the path solution which makes the initial state to the goal method.
        Maintain a list as a queue, named as open_list, append the initial state.
        Always visit and pop the 0 index element, invoke get_possible_moves method find all the possible directions.
        Traversal the possible_moves list and invoke move method to get several new states.Then append them.
        redo the above steps until the open_list is empty or the state has changed to the goal state.
        :return path: list of str, the solution to the goal state.
        >>> eightPuzzle = EightPuzzle([[1, 2, 3], [4, 5, 6], [7, 0, 8]])
        >>> eightPuzzle.solve()
        ['right']
        """
        open_list = deque([(self.initial_state, [])])  
        visited = set([tuple(self.initial_state)])

        while open_list:
            current_state, path = open_list.popleft()
            if current_state == self.goal_state:
                return path

            for move in self.get_possible_moves(current_state):
                new_state = self.move(current_state, move)
                if tuple(new_state) not in visited:
                    visited.add(tuple(new_state))
                    open_list.append((new_state, path + [move]))

        return None  # No solution found



if __name__ == "__main__":
    eightPuzzle = EightPuzzle([[2, 3, 4], [5, 8, 1], [6, 0, 7]])
    print(eightPuzzle.find_blank([[2, 3, 4], [5, 8, 1], [6, 0, 7]]))
    print(eightPuzzle.move([[2, 3, 4], [5, 8, 1], [6, 0, 7]], 'left'))
    print(eightPuzzle.get_possible_moves([[2, 3, 4], [5, 8, 1], [6, 0, 7]]))
    print(eightPuzzle.solve())