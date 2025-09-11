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
        total_icons = self.BOARD_SIZE[0] * self.BOARD_SIZE[1] // 2
        icons = random.sample(self.ICONS * 2, total_icons)
        board = []
        
        for i in range(self.BOARD_SIZE[0]):
            row = icons[i * self.BOARD_SIZE[1]:(i + 1) * self.BOARD_SIZE[1]]
            board.append(row)

        return board

    def is_valid_move(self, pos1, pos2):
        """
        check if the move of two icons is valid
        :param pos1: position tuple(x, y) of the first icon
        :param pos2: position tuple(x, y) of the second icon
        :return: True or False
        """
        if pos1 == pos2:
            return False
        
        x1, y1 = pos1
        x2, y2 = pos2

        if not (0 <= x1 < self.BOARD_SIZE[0] and 0 <= y1 < self.BOARD_SIZE[1]):
            return False
        if not (0 <= x2 < self.BOARD_SIZE[0] and 0 <= y2 < self.BOARD_SIZE[1]):
            return False
        if self.board[x1][y1] != self.board[x2][y2]:
            return False
        
        return self.has_path(pos1, pos2)

    def has_path(self, pos1, pos2):
        """
        check if there is a path between two icons
        :param pos1: position tuple(x, y) of the first icon
        :param pos2: position tuple(x, y) of the second icon
        :return: True or False
        """
        # Implementing a simple path finding algorithm (DFS)
        from collections import deque
        
        def valid_neighbors(x, y):
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.BOARD_SIZE[0] and 0 <= ny < self.BOARD_SIZE[1]:
                    neighbors.append((nx, ny))
            return neighbors

        def dfs(start, end, visited):
            if start == end:
                return True
            visited.add(start)
            for neighbor in valid_neighbors(start[0], start[1]):
                if neighbor not in visited and self.board[neighbor[0]][neighbor[1]] == self.board[start[0]][start[1]]:
                    if dfs(neighbor, end, visited):
                        return True
            return False

        return dfs(pos1, pos2, set())

    def remove_icons(self, pos1, pos2):
        """
        remove the connected icons on the game board
        :param pos1: position tuple(x, y) of the first icon to be removed
        :param pos2: position tuple(x, y) of the second icon to be removed
        :return: None
        """
        if self.is_valid_move(pos1, pos2):
            x1, y1 = pos1
            x2, y2 = pos2
            self.board[x1][y1] = ' '
            self.board[x2][y2] = ' '

    def is_game_over(self):
        """
        Check if the game is over
        :return: True or False
        """
        for row in self.board:
            if any(cell != ' ' for cell in row):
                return False
        return True


if __name__ == "__main__":
    mc = MahjongConnect([4, 4], ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    print("Initial board:")
    for row in mc.board:
        print(row)

    print(mc.is_valid_move((0, 0), (1, 0)))

    print(mc.has_path((0, 0), (1, 0)))

    mc.remove_icons((0, 0), (1, 0))
    print("Board after removing icons:")
    for row in mc.board:
        print(row)

    print(mc.is_game_over())