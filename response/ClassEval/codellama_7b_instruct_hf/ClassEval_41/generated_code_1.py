class GomokuGame:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = [[' ' for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = 'X'

    def make_move(self, row, col):
        if self.board[row][col] != ' ':
            return False
        self.board[row][col] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        return True

    def check_winner(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self._check_five_in_a_row(i, j, (1, 0)):
                    return self.current_player
                if self._check_five_in_a_row(i, j, (0, 1)):
                    return self.current_player
                if self._check_five_in_a_row(i, j, (1, 1)):
                    return self.current_player
        return None

    def _check_five_in_a_row(self, row, col, direction):
        count = 1
        while row + direction[0] < self.board_size and col + direction[1] < self.board_size:
            if self.board[row + direction[0]][col + direction[1]] == self.current_player:
                count += 1
                if count == 5:
                    return True
            else:
                count = 1
            row += direction[0]
            col += direction[1]
        return False

if __name__ == "__main__":
    game = GomokuGame(10)
    moves = [(5, 5), (0, 0), (5, 4), (0, 1), (5, 3), (0, 2), (5, 2), (0, 3), (5, 1)]
    for move in moves:
        game.make_move(move[0], move[1])
    print(game.check_winner())