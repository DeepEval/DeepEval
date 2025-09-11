class TicTacToe:
    """
    The class represents a game of Tic-Tac-Toe and its functions include making a move on the board, checking for a winner, and determining if the board is full.
    """

    def __init__(self, N=3):
        """
        Initialize a 3x3 game board with all empty spaces and current symble player, default is 'X'.
        """
        self.board = [[' ' for _ in range(N)] for _ in range(3)]
        self.current_player = 'X'

    def make_move(self, row, col):
        """
        Place the current player's mark at the specified position on the board and switch the mark.
        :param row: int, the row index of the position
        :param col: int, the column index of the position
        :return: bool, indicating whether the move was successful or not
        """
        if self.board[row][col] == ' ':
            self.board[row][col] = self.current_player
            if self.current_player == 'X':
                self.current_player = 'O'
            else:
                self.current_player = 'X'
            return True
        else:
            return False

    def check_winner(self):
        """
        Check if there is a winner on the board in rows, columns and diagonals three directions
        :return: str or None, the mark of the winner ('X' or 'O'), or None if there is no winner yet
        """
        # Check rows and columns
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != ' ':
                return self.board[i][0]
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != ' ':
                return self.board[0][i]

        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]

        return None

    def is_board_full(self):
        """
        Check if the game board is completely filled.
        :return: bool, indicating whether the game board is full or not
        """
        return all(cell != ' ' for row in self.board for cell in row)

# Example usage
if __name__ == "__main__":
    ttt = TicTacToe()

    # Test making a move
    print(ttt.make_move(0, 0))  # 'O'
    print(ttt.make_move(1, 1))  # 'X'
    print(ttt.make_move(0, 1))  # False

    # Test checking for a winner
    ttt.make_move(0, 2)
    ttt.make_move(1, 0)
    ttt.make_move(2, 0)
    winner = ttt.check_winner()
    print(winner)  # 'O'

    # Test if the board is full
    ttt.make_move(2, 1)
    ttt.make_move(1, 2)
    ttt.make_move(0, 2)
    print(ttt.is_board_full())  # False
    ttt.make_move(2, 2)
    print(ttt.is_board_full())  # True