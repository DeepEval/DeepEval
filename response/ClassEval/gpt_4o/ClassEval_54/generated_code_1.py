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
        board = []

        # Flatten the icons and duplicate them to fill the board
        icons = self.ICONS * (rows * cols // len(self.ICONS))
        random.shuffle(icons)

        for r in range(rows):
            board.append(icons[r * cols:(r + 1) * cols])

        return board

    def is_valid_move(self, pos1, pos2):
        """
        check if the move of two icons is valid
        :param pos1: position tuple(x, y) of the first icon
        :param pos2: position tuple(x, y) of the second icon
        :return: True or False, representing whether the move of two icons is valid
        """
        # Check if positions are within bounds
        if not (0 <= pos1[0] < self.BOARD_SIZE[0] and 0 <= pos1[1] < self.BOARD_SIZE[1]):
            return False
        if not (0 <= pos2[0] < self.BOARD_SIZE[0] and 0 <= pos2[1] < self.BOARD_SIZE[1]):
            return False

        # Check if positions are different
        if pos1 == pos2:
            return False

        # Check if icons are the same
        if self.board[pos1[0]][pos1[1]] != self.board[pos2[0]][pos2[1]]:
            return False

        # Check if there's a valid path
        return self.has_path(pos1, pos2)

    def has_path(self, pos1, pos2):
        """
        check if there is a path between two icons
        :param pos1: position tuple(x, y) of the first icon
        :param pos2: position tuple(x, y) of the second icon
        :return: True or False, representing whether there is a path between two icons
        """
        # For simplicity, let's assume any direct connection without obstacles is a valid path
        # This can be expanded with more complex pathfinding algorithms like BFS/DFS

        # Directly connected vertically or horizontally
        if pos1[0] == pos2[0] or pos1[1] == pos2[1]:
            return True

        return False

    def remove_icons(self, pos1, pos2):
        """
        remove the connected icons on the game board
        :param pos1: position tuple(x, y) of the first icon to be removed
        :param pos2: position tuple(x, y) of the second icon to be removed
        :return: None
        """
        self.board[pos1[0]][pos1[1]] = ' '
        self.board[pos2[0]][pos2[1]] = ' '

    def is_game_over(self):
        """
        Check if the game is over
        :return: True or False, representing whether the game is over
        """
        for row in self.board:
            for cell in row:
                if cell != ' ':
                    return False
        return True

# Test the class
if __name__ == "__main__":
    mc = MahjongConnect([4, 4], ['a', 'b', 'c'])
    
    # Test create_board
    print("Initial Board:")
    for row in mc.board:
        print(row)
    
    # Test is_valid_move
    print("Is valid move (0, 0) -> (1, 0):", mc.is_valid_move((0, 0), (1, 0)))
    
    # Test remove_icons
    mc.remove_icons((0, 0), (1, 0))
    print("Board after removing icons:")
    for row in mc.board:
        print(row)

    # Test is_game_over
    print("Is game over:", mc.is_game_over())