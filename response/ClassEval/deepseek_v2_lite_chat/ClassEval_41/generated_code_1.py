import numpy as np

class GomokuGame:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = np.full((board_size, board_size), ' ')
        self.current_player = 'X'

    def make_move(self, row, col):
        if self.board[row, col] == ' ':
            self.board[row, col] = self.current_player
            if self.check_winner(row, col):
                return False  # Game over
            if self._check_five_in_a_row(row, col, (0, 1)):
                return False  # Game over
            if self.current_player == 'X':
                self.current_player = 'O'
            else:
                self.current_player = 'X'
            return True
        return False

    def check_winner(self, row, col):
        for direction in ((0, 1), (1, 0), (1, 1), (1, -1)):
            if self._check_five_in_a_row(row, col, direction):
                return self.board[row, col]
        return None

    def _check_five_in_a_row(self, row, col, direction):
        count = 0
        for i, j in zip(direction, direction[1:]):
            i = row + i[0]
            j = col + i[1]
            if 0 <= i < self.board_size and 0 <= j < self.board_size and self.board[i, j] == self.current_player:
                count += 1
                if count == 5:
                    return True
            else:
                count = 0
        return False

# Test cases
if __name__ == "__main__":
    gomoku_game = GomokuGame(10)
    # Test make_move
    print(gomoku_game.make_move(5, 5))  # True
    print(gomoku_game.make_move(5, 5))  # False
    # Test check_winner
    gomoku_game.make_move(5, 5)
    gomoku_game.make_move(0, 0)
    gomoku_game.make_move(5, 4)
    gomoku_game.make_move(0, 1)
    gomoku_game.make_move(5, 3)
    gomoku_game.make_move(0, 2)
    gomoku_game.make_move(5, 2)
    gomoku_game.make_move(0, 3)
    gomoku_game.make_move(5, 1)
    print(gomoku_game.check_winner())  # 'X'
    # Test _check_five_in_a_row
    moves = [(5, 5), (0, 0), (5, 4), (0, 1), (5, 3), (0, 2), (5, 2), (0, 3), (5, 1)]
    for move in moves:
        gomoku_game.make_move(move[0], move[1])
    print(gomoku_game._check_five_in_a_row(5, 1, (0, 1)))  # True
    print(gomoku_game._check_five_in_a_row(5, 1, (1, 1)))  # False