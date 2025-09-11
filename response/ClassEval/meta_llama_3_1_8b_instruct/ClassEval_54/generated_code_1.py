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
        board = []
        for _ in range(self.BOARD_SIZE[0]):
            row = []
            for _ in range(self.BOARD_SIZE[1]):
                row.append(random.choice(self.ICONS))
            board.append(row)
        return board

    def is_valid_move(self, pos1, pos2):
        """
        check if the move of two icons is valid (i.e. positions are within the game board range, the two positions are not the same, the two positions have the same icon, and there is a valid path between the two positions)
        :param pos1: position tuple(x, y) of the first icon
        :param pos2: position tuple(x, y) of the second icon
        :return:True or False,representing whether the move of two icons is valid
        >>> mc = MahjongConnect([4, 4], ['a', 'b', 'c'])
        mc.board = [['a', 'b', 'c', 'a'],
                    ['a', 'b', 'c', 'a'],
                    ['a', 'b', 'c', 'a'],
                    ['a', 'b', 'c', 'a']]
        >>> mc.is_valid_move((0, 0), (1, 0))
        True
        """
        if pos1 == pos2:
            return False
        if (pos1[0] < 0 or pos1[0] >= self.BOARD_SIZE[0] or
            pos1[1] < 0 or pos1[1] >= self.BOARD_SIZE[1] or
            pos2[0] < 0 or pos2[0] >= self.BOARD_SIZE[0] or
            pos2[1] < 0 or pos2[1] >= self.BOARD_SIZE[1]):
            return False
        if self.board[pos1[0]][pos1[1]]!= self.board[pos2[0]][pos2[1]]:
            return False
        return self.has_path(pos1, pos2)

    def has_path(self, pos1, pos2):
        """
        check if there is a path between two icons
        :param pos1: position tuple(x, y) of the first icon
        :param pos2: position tuple(x, y) of the second icon
        :return: True or False,representing whether there is a path between two icons
        >>> mc = MahjongConnect([4, 4], ['a', 'b', 'c'])
        mc.board = [['a', 'b', 'c', 'a'],
                    ['a', 'b', 'c', 'a'],
                    ['a', 'b', 'c', 'a'],
                    ['a', 'b', 'c', 'a']]
        >>> mc.has_path((0, 0), (1, 0))
        True
        """
        x_diff = abs(pos1[0] - pos2[0])
        y_diff = abs(pos1[1] - pos2[1])

        # if the two positions are adjacent
        if (x_diff == 1 and y_diff == 0) or (x_diff == 0 and y_diff == 1):
            return True

        # if the two positions are not adjacent, check if there is a path
        if x_diff!= 0 and y_diff!= 0:
            return False

        dx, dy = 1, 0
        if pos1[0] > pos2[0]:
            dx = -1
        if pos1[1] > pos2[1]:
            dy = -1
        x, y = pos1[0], pos1[1]
        while (x, y)!= pos2:
            if self.board[x][y]!= self.board[pos1[0]][pos1[1]]:
                return False
            x += dx
            y += dy
        return True

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
        if not self.is_valid_move(pos1, pos2):
            return
        x, y = pos1[0], pos1[1]
        dx, dy = 1, 0
        if pos1[0] > pos2[0]:
            dx = -1
        if pos1[1] > pos2[1]:
            dy = -1
        while (x, y)!= pos2:
            self.board[x][y] =''
            x += dx
            y += dy

    def is_game_over(self):
        """
        Check if the game is over (i.e., if there are no more icons on the game board)
        :return: True or False,representing whether the game is over
        >>> mc = MahjongConnect([4, 4], ['a', 'b', 'c'])
        >>> mc.board = [[' ','','',''],
        >>>         [' ','','',''],
        >>>         [' ','','',''],
        >>>         [' ','','','']]
        >>> mc.is_game_over()
        True
        """
        return all(all(icon =='' for icon in row) for row in self.board)


if __name__ == "__main__":
    mc = MahjongConnect([4, 4], ['a', 'b', 'c'])
    print("Initial Board:")
    for row in mc.board:
        print(row)
    print("\nIs valid move from (0, 0) to (1, 0)?", mc.is_valid_move((0, 0), (1, 0)))
    mc.remove_icons((0, 0), (1, 0))
    print("\nBoard after removing icons:")
    for row in mc.board:
        print(row)
    print("\nIs game over?", mc.is_game_over())