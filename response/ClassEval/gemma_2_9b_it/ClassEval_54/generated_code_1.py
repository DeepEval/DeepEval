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
        >>>mc = MahjongConnect([4, 4], ['a', 'b', 'c'])
        mc.BOARD_SIZE = [4, 4]
        mc.ICONS = ['a', 'b', 'c']
        mc.board = mc.create_board()
        """
        self.BOARD_SIZE = BOARD_SIZE
        self.ICONS = ICONS
        self.board = self.create_board()

    def create_board(self):
        """
        create the game board with the given board size and icons
        :return: 2-dimensional list, the game board
        >>> mc = MahjongConnect([4, 4], ['a', 'b', 'c'])
        >>> mc.create_board()
        mc.board = [['a', 'b', 'c', 'a'],
                    ['a', 'b', 'c', 'a'],
                    ['a', 'b', 'c', 'a'],
                    ['a', 'b', 'c', 'a']]
        """
        board = [['' for _ in range(self.BOARD_SIZE[1])] for _ in range(self.BOARD_SIZE[0])]
        for i in range(self.BOARD_SIZE[0]):
            for j in range(self.BOARD_SIZE[1]):
                board[i][j] = random.choice(self.ICONS)
        return board

    def is_valid_move(self, pos1, pos2):
        """
        check if the move of two icons is valid (i.e. positions are within the game board range, the two positions are not the same, the two positions have the same icon, and there is a valid path between the two positions)
        :param pos1: position tuple(x, y) of the first icon
        :param pos2: position tuple(x, y) of the second icon
        :return:True or False ,representing whether the move of two icons is valid
        >>> mc = MahjongConnect([4, 4], ['a', 'b', 'c'])
        mc.board = [['a', 'b', 'c', 'a'],
                    ['a', 'b', 'c', 'a'],
                    ['a', 'b', 'c', 'a'],
                    ['a', 'b', 'c', 'a']]
        >>> mc.is_valid_move((0, 0), (1, 0))
        True
        """
        x1, y1 = pos1
        x2, y2 = pos2
        if x1 == x2 and y1 == y2:
            return False
        if 0 <= x1 < self.BOARD_SIZE[0] and 0 <= y1 < self.BOARD_SIZE[1] and 0 <= x2 < self.BOARD_SIZE[0] and 0 <= y2 < self.BOARD_SIZE[1] and self.board[x1][y1] == self.board[x2][y2]:
            return self.has_path(pos1, pos2)
        return False

    def has_path(self, pos1, pos2):
        """
        check if there is a path between two icons
        :param pos1: position tuple(x, y) of the first icon
        :param pos2: position tuple(x, y) of the second icon
        :return: True or False ,representing whether there is a path between two icons
        >>> mc = MahjongConnect([4, 4], ['a', 'b', 'c'])
        mc.board = [['a', 'b', 'c', 'a'],
                    ['a', 'b', 'c', 'a'],
                    ['a', 'b', 'c', 'a'],
                    ['a', 'b', 'c', 'a']]
        >>> mc.is_valid_move((0, 0), (1, 0))
        True
        """
        x1, y1 = pos1
        x2, y2 = pos2
        if x1 == x2 and y1 == y2:
            return True
        if x1 < 0 or x1 >= self.BOARD_SIZE[0] or y1 < 0 or y1 >= self.BOARD_SIZE[1] or x2 < 0 or x2 >= self.BOARD_SIZE[0] or y2 < 0 or y2 >= self.BOARD_SIZE[1]:
            return False
        if self.board[x1][y1] != self.board[x2][y2]:
            return False
        return self.has_path((x1, y1 - 1), (x2, y2 - 1)) or self.has_path((x1, y1 + 1), (x2, y2 + 1)) or self.has_path((x1 - 1, y1), (x2 - 1, y2)) or self.has_path((x1 + 1, y1), (x2 + 1, y2))

    def remove_icons(self, pos1, pos2):
        """
        remove the connected icons on the game board
        :param pos1: position tuple(x, y) of the first icon to be removed
        :param pos2: position tuple(x, y) of the second icon to be removed
        :return: None
        >>> mc = MahjongConnect([4, 4], ['a', 'b', 'c'])
        mc.board = [['a', 'b', 'c', 'a'],
                    ['a', 'b', 'c', 'a'],
                    ['a', 'b', 'c', 'a'],
                    ['a', 'b', 'c', 'a']]
        >>> mc.remove_icons((0, 0), (1, 0))
        mc.board = [[' ', 'b', 'c', 'a'],
                    [' ', 'b', 'c', 'a'],
                    ['a', 'b', 'c', 'a'],
                    ['a', 'b', 'c', 'a']]
        """
        x1, y1 = pos1
        x2, y2 = pos2
        self.board[x1][y1] = ' '
        self.board[x2][y2] = ' '

    def is_game_over(self):
        """
        Check if the game is over (i.e., if there are no more icons on the game board)
        :return: True or False ,representing whether the game is over
        >>> mc = MahjongConnect([4, 4] ['a', 'b', 'c'])
        >>> mc.board = [[' ', ' ', ' ', ' '],
        >>>         [' ', ' ', ' ', ' '],
        >>>         [' ', ' ', ' ', ' '],
        >>>         [' ', ' ', ' ', ' ']]
        >>> mc.is_game_over()
        True
        """
        for row in self.board:
            for cell in row:
                if cell != ' ':
                    return False
        return True

if __name__ == "__main__":
    instance = MahjongConnect([4, 4], ['a', 'b', 'c'])
    # Test case for create_board
    output = instance.create_board()
    print(output)

    # Test case for is_valid_move
    output = instance.is_valid_move((0, 0), (1, 0))
    print(output)

    # Test case for has_path
    output = instance.has_path((0, 0), (1, 0))
    print(output)

    # Test case for remove_icons
    instance.remove_icons((0, 0), (1, 0))
    print(instance.board)

    # Test case for is_game_over
    instance.board = [[' ', ' ', ' ', ' '],
                    [' ', ' ', ' ', ' '],
                    [' ', ' ', ' ', ' '],
                    [' ', ' ', ' ', ' ']]
    output = instance.is_game_over()
    print(output)