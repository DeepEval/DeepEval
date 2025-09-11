class GomokuGame:
    """
    This class is an implementation of a Gomoku game, supporting for making moves, checking for a winner,
    and checking if there are five consecutive symbols on the game board.
    """

    def __init__(self, board_size):
        """
        Initializes the game with a given board size.
        It initializes the board with empty spaces and sets the current player symbol as 'X'.
        """
        self.board_size = board_size
        self.board = [[' ' for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = 'X'

    def make_move(self, row, col):
        """
        Makes a move at the given row and column.
        If the move is valid, it places the current player's symbol on the board
        and changes the current player to the other player.
        :param row: int, the row index of this move
        :param col: int, the column index
        return: True if the move is valid, or False otherwise.
        """
        if 0 <= row < self.board_size and 0 <= col < self.board_size and self.board[row][col] == ' ':
            self.board[row][col] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False

    def check_winner(self):
        """
        Checks if there is a winner by looking for five in a row in all directions (horizontal, vertical, diagonal).
        return: the symbol of the winning player (either 'X' or 'O') if there is a winner, or None otherwise.
        """
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] != ' ':
                    if (self._check_five_in_a_row(row, col, (0, 1)) or  # Horizontal
                        self._check_five_in_a_row(row, col, (1, 0)) or  # Vertical
                        self._check_five_in_a_row(row, col, (1, 1)) or  # Diagonal \
                        self._check_five_in_a_row(row, col, (1, -1))):  # Diagonal /
                        return self.board[row][col]
        return None

    def _check_five_in_a_row(self, row, col, direction):
        """
        Checks if there are five consecutive symbols of the same player in a row starting from a given cell in a given direction.
        :param row: int, row of the given cell
        :param col: int, column of the given cell
        :param direction: tuple, (int, int), named as (dx, dy).
        :return: True if there are five consecutive symbols of the same player, and False otherwise.
        """
        count = 0
        dx, dy = direction
        player_symbol = self.board[row][col]

        for i in range(5):
            new_row = row + i * dx
            new_col = col + i * dy
            if 0 <= new_row < self.board_size and 0 <= new_col < self.board_size:
                if self.board[new_row][new_col] == player_symbol:
                    count += 1
                else:
                    break
            else:
                break

        return count == 5


if __name__ == "__main__":
    # Test case for __init__
    game = GomokuGame(10)
    print(game.board)  # Should print a 10x10 board of spaces

    # Test case for make_move
    output = game.make_move(5, 5)
    print(output)  # True
    output = game.make_move(5, 5)
    print(output)  # False

    # Test case for check_winner
    moves = [(5, 5), (0, 0), (5, 4), (0, 1), (5, 3), (0, 2), (5, 2), (0, 3), (5, 1)]
    for move in moves:
        game.make_move(move[0], move[1])
    winner = game.check_winner()
    print(winner)  # 'X', since 'X' will have five in a row vertically

    # Test case for _check_five_in_a_row
    result = game._check_five_in_a_row(5, 1, (0, 1))
    print(result)  # True, there are five 'X' in a row horizontally from (5, 1)
    result = game._check_five_in_a_row(5, 1, (1, 1))
    print(result)  # False