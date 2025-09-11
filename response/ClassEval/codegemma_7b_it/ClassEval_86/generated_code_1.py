class TicTacToe:
    """
    The class represents a game of Tic-Tac-Toe and its functions include making a move on the board, checking for a winner, and determining if the board is full.
    """

    def __init__(self, N=3):
        """
        Initialize a 3x3 game board with all empty spaces and current symble player, default is 'X'.
        """
        self.board = [[' ' for _ in range(N)] for _ in range(N)]
        self.current_player = 'X'

    def make_move(self, row, col):
        """
        Place the current player's mark at the specified position on the board and switch the mark.
        :param row: int, the row index of the position
        :param col: int, the column index of the position
        :return: bool, indicating whether the move was successful or not
        """
        if row < 0 or row >= len(self.board) or col < 0 or col >= len(self.board[0]) or self.board[row][col] != ' ':
            return False
        self.board[row][col] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        return True

    def check_winner(self):
        """
        Check if there is a winner on the board in rows, columns and diagonals three directions
        :return: str or None, the mark of the winner ('X' or 'O'), or None if there is no winner yet
        """
        for i in range(len(self.board)):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != ' ':
                return self.board[i][0]
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != ' ':
                return self.board[0][i]
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
        for row in self.board:
            if ' ' in row:
                return False
        return True

if __name__ == "__main__":
    ttt = TicTacToe()
    # Test case for make_move()
    print(ttt.make_move(1, 1))  # True
    print(ttt.board)
    print(ttt.current_player)  # O
    print(ttt.make_move(1, 1))  # False
    print(ttt.make_move(1, 0))  # True
    print(ttt.board)
    print(ttt.current_player)  # X

    # Test case for check_winner()
    moves = [(1, 0), (2, 0), (1, 1), (2, 1), (1, 2)]
    for move in moves:
        ttt.make_move(move[0], move[1])
    print(ttt.check_winner())  # X

    # Test case for is_board_full()
    print(ttt.is_board_full())  # False
    ttt.make_move(0, 0)
    print(ttt.is_board_full())  # True