class TicTacToe:
    """
    The class represents a game of Tic-Tac-Toe and its functions include making
    a move on the board, checking for a winner, and determining if the board is full.
    """

    def __init__(self, N=3):
        """
        Initialize a 3x3 game board with all empty spaces and current symbol player,
        default is 'X'.
        """
        self.board = [['' for _ in range(N)] for _ in range(3)]
        self.current_player = 'X'

    def make_move(self, row, col):
        """
        Place the current player's mark at the specified position on the board
        and switch the mark.
        :param row: int, the row index of the position
        :param col: int, the column index of the position
        :return: bool, indicating whether the move was successful or not
        >>> ttt.make_move(1, 1)
        >>> ttt.board[1][1]
        'X'
        >>> ttt.make_move(1, 1)
        False
        """
        if self.board[row][col] == '':
            self.board[row][col] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        else:
            return False

    def check_winner(self):
        """
        Check if there is a winner on the board in rows, columns and diagonals
        three directions
        :return: str or None, the mark of the winner ('X' or 'O'), or None
        if there is no winner yet
        >>> moves = [(0, 0), (0, 1), (0, 2)]
        >>> for move in moves:
       ...     ttt.make_move(move[0], move[1])
        >>> ttt.check_winner()
        'X'
        """
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2]!= '':
                return row[0]

        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col]!= '':
                return self.board[0][col]

        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2]!= '':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0]!= '':
            return self.board[0][2]

        return None

    def is_board_full(self):
        """
        Check if the game board is completely filled.
        :return: bool, indicating whether the game board is full or not
        >>> ttt.is_board_full()
        False
        """
        return all(cell!= '' for row in self.board for cell in row)


if __name__ == "__main__":
    ttt = TicTacToe()
    print("Initial Board:")
    for row in ttt.board:
        print(row)

    # Test make_move
    print("\nTest make_move:")
    print(ttt.make_move(1, 1))
    print(ttt.board[1][1])

    # Test check_winner
    print("\nTest check_winner:")
    moves = [(0, 0), (0, 1), (0, 2)]
    for move in moves:
        ttt.make_move(move[0], move[1])
    print(ttt.check_winner())

    # Test is_board_full
    print("\nTest is_board_full:")
    print(ttt.is_board_full())