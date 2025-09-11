import random

class MahjongConnect:
    """
    MahjongConnect is a class representing a game board for Mahjong Connect with features like creating the board, checking valid moves, finding paths, removing icons, and checking if the game is over.
    """

    def __init__(self, BOARD_SIZE, ICONS):
        """
        Initialize the board size and the icon list, create the game board
        :param BOARD_SIZE: list of two integer numbers, representing the number of rows and columns of the game board
        :param ICONS: list of string, representing the icons
        """
        self.BOARD_SIZE = BOARD_SIZE
        self.ICONS = ICONS
        self.board = self.create_board()

    def create_board(self):
        """
        Create the game board with the given board size and icons
        :return: 2-dimensional list, the game board
        """
        return [[' ' for _ in range(self.BOARD_SIZE[0])] for _ in range(self.BOARD_SIZE[1])]

    def is_valid_move(self, pos1, pos2):
        """
        Check if the move of two icons is valid (i.e., positions are within the game board range, the two positions have the same icon, and there is a valid path between the two positions)
        :param pos1: Position tuple(x, y) of the first icon
        :param pos2: Position tuple(x, y) of the second icon
        :return: True or False, representing whether the move of two icons is valid
        """
        # Implement the logic to check if the move is valid
        pass

    def has_path(self, pos1, pos2):
        """
        Check if there is a path between two icons
        :param pos1: Position tuple(x, y) of the first icon
        :param pos2: Position tuple(x, y) of the second icon
        :return: True or False, representing whether there is a path between two icons
        """
        # Implement the logic to check if there is a path between two icons
        pass

    def remove_icons(self, pos1, pos2):
        """
        Remove the connected icons on the game board
        :param pos1: Position tuple(x, y) of the first icon to be removed
        :param pos2: Position tuple(x, y) of the second icon to be removed
        :return: None
        """
        # Implement the logic to remove connected icons
        pass

    def is_game_over(self):
        """
        Check if the game is over (i.e., if there are no more icons on the game board)
        :return: True or False, representing whether the game is over
        """
        # Implement the logic to check if the game is over
        pass

# Test cases
if __name__ == "__main__":
    mc = MahjongConnect([4, 4], ['a', 'b', 'c'])
    print(mc.create_board())
    print(mc.is_valid_move((0, 0), (1, 0)))
    print(mc.has_path((0, 0), (1, 0)))
    mc.remove_icons((0, 0), (1, 0))
    print(mc.is_game_over())