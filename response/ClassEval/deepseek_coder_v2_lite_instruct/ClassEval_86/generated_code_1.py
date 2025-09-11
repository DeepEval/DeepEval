class TicTacToe:
    """
    The class represents a game of Tic-Tac-Toe and its functions include making a move on the board, checking for a winner, and determining if the board is full.
    """

    def __init__(self, N=3):
        """
        Initialize a 3x3 game board with all empty spaces and current symbol player, default is 'X'.
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
        if self.board[row][col] == ' ':
            self.board[row][col] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False

    def check_winner(self):
        """
        Check if there is a winner on the board in rows, columns and diagonals three directions
        :return: str or None, the mark of the winner ('X' or 'O'), or None if there is no winner yet
        """
        N = len(self.board)
        # Check rows and columns
        for i in range(N):
            if all(self.board[i][j] == self.board[i][0] and self.board[i][0] != ' ' for j in range(N)):
                return self.board[i][0]
            if all(self.board[j][i] == self.board[0][i] and self.board[0][i] != ' ' for j in range(N)):
                return self.board[0][i]
        # Check diagonals
        if all(self.board[i][i] == self.board[0][0] and self.board[0][0] != ' ' for i in range(N)):
            return self.board[0][0]
        if all(self.board[i][N-1-i] == self.board[0][N-1] and self.board[0][N-1] != ' ' for i in range(N)):
            return self.board[0][N-1]
        return None

    def is_board_full(self):
        """
        Check if the game board is completely filled.
        :return: bool, indicating whether the game board is full or not
        """
        return all(self.board[i][j] != ' ' for i in range(len(self.board)) for j in range(len(self.board)))

if __name__ == "__main__":
    ttt = TicTacToe()
    
    # Test make_move
    print("Make move (0, 0):", ttt.make_move(0, 0))  # True
    print("Current player after move:", ttt.current_player)  # 'O'
    
    # Test check_winner
    moves = [(1, 0), (2, 0), (1, 1), (2, 1), (1, 2)]
    for move in moves:
        ttt.make_move(move[0], move[1])
    print("Winner:", ttt.check_winner())  # 'X'
    
    # Test is_board_full
    print("Board full:", ttt.is_board_full())  # False
    ttt.make_move(2, 2)
    ttt.make_move(0, 1)
    ttt.make_move(1, 1)
    ttt.make_move(0, 2)
    print("Board full:", ttt.is_board_full())  # True