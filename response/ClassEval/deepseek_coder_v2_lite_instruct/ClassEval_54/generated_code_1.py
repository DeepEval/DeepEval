import random

class MahjongConnect:
    """
    MahjongConnect is a class representing a game board for Mahjong Connect with features like creating the board, checking valid moves, finding paths, removing icons, and checking if the game is over.
    """

    def __init__(self, BOARD_SIZE, ICONS):
        """
        initialize the board size and the icon list, create the game board
        :param BOARD_SIZE: list of two integer numbers, representing the number of rows and columns of the game board
        :param ICONS: list of string, representing the icons
        """
        self.BOARD_SIZE = BOARD_SIZE
        self.ICONS = ICONS
        self.board = self.create_board()

    def create_board(self):
        """
        create the game board with the given board size and icons
        :return: 2-dimensional list, the game board
        """
        rows, cols = self.BOARD_SIZE
        board = [[random.choice(self.ICONS) for _ in range(cols)] for _ in range(rows)]
        return board

    def is_valid_move(self, pos1, pos2):
        """
        check if the move of two icons is valid (i.e. positions are within the game board range, the two positions are not the same, the two positions have the same icon, and there is a valid path between the two positions)
        :param pos1: position tuple(x, y) of the first icon
        :param pos2: position tuple(x, y) of the second icon
        :return: True or False, representing whether the move of two icons is valid
        """
        x1, y1 = pos1
        x2, y2 = pos2
        if not (0 <= x1 < self.BOARD_SIZE[0] and 0 <= y1 < self.BOARD_SIZE[1]):
            return False
        if not (0 <= x2 < self.BOARD_SIZE[0] and 0 <= y2 < self.BOARD_SIZE[1]):
            return False
        if pos1 == pos2:
            return False
        if self.board[x1][y1] != self.board[x2][y2]:
            return False
        if not self.has_path(pos1, pos2):
            return False
        return True

    def has_path(self, pos1, pos2):
        """
        check if there is a path between two icons
        :param pos1: position tuple(x, y) of the first icon
        :param pos2: position tuple(x, y) of the second icon
        :return: True or False, representing whether there is a path between two icons
        """
        x1, y1 = pos1
        x2, y2 = pos2
        visited = set()
        return self._dfs(x1, y1, x2, y2, visited)

    def _dfs(self, x1, y1, x2, y2, visited):
        if (x1, y1) == (x2, y2):
            return True
        if (x1, y1) in visited:
            return False
        visited.add((x1, y1))
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            nx, ny = x1 + dx, y1 + dy
            if 0 <= nx < self.BOARD_SIZE[0] and 0 <= ny < self.BOARD_SIZE[1] and self._dfs(nx, ny, x2, y2, visited):
                return True
        return False

    def remove_icons(self, pos1, pos2):
        """
        remove the connected icons on the game board
        :param pos1: position tuple(x, y) of the first icon to be removed
        :param pos2: position tuple(x, y) of the second icon to be removed
        :return: None
        """
        x1, y1 = pos1
        x2, y2 = pos2
        icon = self.board[x1][y1]
        self.board[x1][y1] = ' '
        self.board[x2][y2] = ' '
        self._clear_empty_rows_and_cols()

    def _clear_empty_rows_and_cols(self):
        rows, cols = self.BOARD_SIZE
        # Clear empty rows
        self.board = [row for row in self.board if any(cell != ' ' for cell in row)]
        # Clear empty columns
        self.board = [[row[col] for row in self.board if row[col] != ' '] for col in range(cols)]
        # Fill empty rows with new icons
        new_board = []
        for row in self.board:
            if len(row) < cols:
                row.extend([random.choice(self.ICONS) for _ in range(cols - len(row))])
            new_board.append(row)
        self.board = new_board

    def is_game_over(self):
        """
        Check if the game is over (i.e., if there are no more icons on the game board)
        :return: True or False, representing whether the game is over
        """
        return all(cell == ' ' for row in self.board for cell in row)

# Test cases
if __name__ == "__main__":
    mc = MahjongConnect([4, 4], ['a', 'b', 'c'])
    print("Initial Board:")
    print(mc.board)

    print("\nTesting is_valid_move:")
    print(mc.is_valid_move((0, 0), (1, 0)))  # True
    print(mc.is_valid_move((0, 0), (0, 1)))  # True
    print(mc.is_valid_move((0, 0), (1, 1)))  # False (no path)
    print(mc.is_valid_move((0, 0), (2, 2)))  # False (out of bounds)
    print(mc.is_valid_move((0, 0), (0, 0)))  # False (same position)
    print(mc.is_valid_move((0, 0), (0, 2)))  # False (different icons)

    print("\nTesting has_path:")
    print(mc.has_path((0, 0), (1, 0)))  # True
    print(mc.has_path((0, 0), (0, 1)))  # True
    print(mc.has_path((0, 0), (1, 1)))  # False

    print("\nTesting remove_icons:")
    mc.remove_icons((0, 0), (1, 0))
    print("Board after removing (0, 0) and (1, 0):")
    print(mc.board)

    print("\nTesting is_game_over:")
    print(mc.is_game_over())  # False
    mc.board = [[' ' for _ in range(4)] for _ in range(4)]
    print(mc.is_game_over())  # True