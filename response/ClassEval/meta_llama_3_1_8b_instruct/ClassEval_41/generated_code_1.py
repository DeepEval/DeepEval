class GomokuGame:
    """
    This class is an implementation of a Gomoku game, supporting for making moves, checking for a winner, and checking if there are five consecutive symbols on the game board.
    """

    def __init__(self, board_size):
        """
        Initializes the game with a given board size.
        It initializes the board with empty spaces and sets the current player symble as 'X'.
        """
        self.board_size = board_size
        self.board = [[''for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = 'X'

    def make_move(self, row, col):
        """
        Makes a move at the given row and column.
        If the move is valid, it places the current player's symbol on the board
        and changes the current player to the other player (if the current player is 'X', then it becomes 'O' and vice versa).
        :param row: int, the row index of this move
        :param col: int, the column index
        return: True if the move is valid, or False otherwise.
        """
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size or self.board[row][col]!='':
            return False
        self.board[row][col] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        return True

    def check_winner(self):
        """
        Checks if there is a winner by looking for five in a row in all directions (horizontal, vertical, diagonal).
        return: the symbol of the winning player (either 'X' or 'O') if there is a winner, or None otherwise.
        """
        for row in range(self.board_size):
            for col in range(self.board_size):
                # Check horizontal
                if self._check_five_in_a_row(row, col, (0, 1)):
                    return self.board[row][col]
                # Check vertical
                if self._check_five_in_a_row(row, col, (1, 0)):
                    return self.board[row][col]
                # Check diagonal (top-left to bottom-right)
                if self._check_five_in_a_row(row, col, (1, 1)):
                    return self.board[row][col]
                # Check diagonal (bottom-left to top-right)
                if self._check_five_in_a_row(row, col, (1, -1)):
                    return self.board[row][col]
        return None

    def _check_five_in_a_row(self, row, col, direction):
        """
        checks if there are five consecutive symbols of the same player in a row starting from a given cell in a given direction (horizontal, vertical, diagonal).
        Counts the number of consecutive symbols in that direction starting from the given cell,
        :param row: int, row of the given cell
        :param col: int, column of the given cell
        :param direction: tuple, (int, int), named as (dx, dy). Row and col will plus several dx and dy repectively.
        :return: True if there are five consecutive symbols of the same player, and False otherwise.
        """
        count = 1
        r, c = row + direction[0], col + direction[1]
        while r >= 0 and r < self.board_size and c >= 0 and c < self.board_size and self.board[r][c]!='':
            if self.board[r][c] == self.board[row][col]:
                count += 1
                r += direction[0]
                c += direction[1]
            else:
                break
        r, c = row - direction[0], col - direction[1]
        while r >= 0 and r < self.board_size and c >= 0 and c < self.board_size and self.board[r][c]!='':
            if self.board[r][c] == self.board[row][col]:
                count += 1
                r -= direction[0]
                c -= direction[1]
            else:
                break
        return count >= 5

if __name__ == "__main__":
    # Test case for make_move method
    gomokuGame = GomokuGame(10)
    print(gomokuGame.make_move(5, 5))  # True
    print(gomokuGame.make_move(5, 5))  # False

    # Test case for check_winner method
    gomokuGame = GomokuGame(10)
    moves = [(5, 5), (0, 0), (5, 4), (0, 1), (5, 3), (0, 2), (5, 2), (0, 3), (5, 1)]
    for move in moves:
        gomokuGame.make_move(move[0], move[1])
    print(gomokuGame.check_winner())  # 'X'

    # Test case for _check_five_in_a_row method
    gomokuGame = GomokuGame(10)
    moves = [(5, 5), (0, 0), (5, 4), (0, 1), (5, 3), (0, 2), (5, 2), (0, 3), (5, 1)]
    for move in moves:
        gomokuGame.make_move(move[0], move[1])
    print(gomokuGame._check_five_in_a_row(5, 1, (0, 1)))  # True
    print(gomokuGame._check_five_in_a_row(5, 1, (1, 1)))  # False