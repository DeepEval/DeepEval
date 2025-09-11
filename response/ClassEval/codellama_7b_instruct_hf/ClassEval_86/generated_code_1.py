import random

class TicTacToe:
    def __init__(self, N=3):
        self.board = [[' ' for _ in range(N)] for _ in range(3)]
        self.current_player = 'X'

    def make_move(self, row, col):
        if self.board[row][col] == ' ':
            self.board[row][col] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        else:
            return False

    def check_winner(self):
        rows = [self.board[i] for i in range(len(self.board))]
        cols = [[self.board[j][i] for j in range(len(self.board))] for i in range(len(self.board))]
        diags = [self.board[i][i] for i in range(len(self.board))] + [self.board[i][len(self.board) - i - 1] for i in range(len(self.board))]
        for line in rows + cols + diags:
            if len(line) > 0 and all(cell == line[0] and cell != ' ' for cell in line):
                return line[0]
        return None

    def is_board_full(self):
        return all(all(cell != ' ' for cell in row) for row in self.board)

if __name__ == "__main__":
    ttt = TicTacToe()
    print(ttt.check_winner())
    print(ttt.is_board_full())

    # Test case 1:
    ttt = TicTacToe()
    for i in range(9):
        ttt.make_move(i % 3, i // 3)
    print(ttt.check_winner())

    # Test case 2:
    ttt = TicTacToe()
    for i in range(9):
        ttt.make_move(i % 3, i // 3)
    print(ttt.is_board_full())

    # Test case 3:
    ttt = TicTacToe()
    for i in range(5):
        ttt.make_move(i % 3, i // 3)
    print(ttt.check_winner())

    # Test case 4:
    ttt = TicTacToe()
    for i in range(5):
        ttt.make_move(i % 3, i // 3)
    print(ttt.is_board_full())

    # Test case 5:
    ttt = TicTacToe()
    for i in range(5):
        ttt.make_move(i % 3, i // 3)
    print(ttt.check_winner())

    # Test case 6:
    ttt = TicTacToe()
    for i in range(5):
        ttt.make_move(i % 3, i // 3)
    print(ttt.is_board_full())