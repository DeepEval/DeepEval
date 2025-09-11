import collections

class EightPuzzle:
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        self.open_list = collections.deque([initial_state])
        self.closed_list = []

    def find_blank(self, state):
        for i, row in enumerate(state):
            for j, tile in enumerate(row):
                if tile == 0:
                    return i, j

    def move(self, state, direction):
        blank_pos = self.find_blank(state)
        if direction == 'up':
            new_state = [[tile if i != blank_pos[0] else 0 for i, tile in enumerate(row)] for row in state]
        elif direction == 'down':
            new_state = [[tile if i != blank_pos[0] else 0 for i, tile in enumerate(row)] for row in state]
        elif direction == 'left':
            new_state = [[tile if i != blank_pos[1] else 0 for i, tile in enumerate(row)] for row in state]
        elif direction == 'right':
            new_state = [[tile if i != blank_pos[1] else 0 for i, tile in enumerate(row)] for row in state]
        return new_state

    def get_possible_moves(self, state):
        blank_pos = self.find_blank(state)
        possible_moves = []
        if blank_pos[0] > 0:
            possible_moves.append('up')
        if blank_pos[0] < 2:
            possible_moves.append('down')
        if blank_pos[1] > 0:
            possible_moves.append('left')
        if blank_pos[1] < 2:
            possible_moves.append('right')
        return possible_moves

    def solve(self):
        while self.open_list:
            current_state = self.open_list.popleft()
            if current_state == self.goal_state:
                return 'Solved!'
            for direction in self.get_possible_moves(current_state):
                new_state = self.move(current_state, direction)
                if new_state not in self.closed_list:
                    self.open_list.append(new_state)
                    self.closed_list.append(new_state)
        return 'Unsolved!'
    
if __name__ == "__main__":

    # Test cases
    eight_puzzle = EightPuzzle([[2, 3, 4], [5, 8, 1], [6, 0, 7]])
    print(eight_puzzle.solve()) # Should print "Solved!"

    eight_puzzle = EightPuzzle([[1, 2, 3], [4, 5, 6], [7, 0, 8]])
    print(eight_puzzle.solve()) # Should print "Unsolved!"

    eight_puzzle = EightPuzzle([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    print(eight_puzzle.solve()) # Should print "Solved!"