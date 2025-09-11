import random

class MinesweeperGame:
    """
    This is a class that implements mine sweeping games including minesweeping and winning judgment.
    """

    def __init__(self, n, k) -> None:
        """
        Initializes the MinesweeperGame class with the size of the board and the number of mines.
        :param n: The size of the board, int.
        :param k: The number of mines, int.
        """
        self.n = n
        self.k = k
        self.minesweeper_map = self.generate_mine_sweeper_map()
        self.player_map = self.generate_playerMap()
        self.score = 0

    def generate_mine_sweeper_map(self):
        """
        Generates a minesweeper map with the given size of the board and the number of mines,the given parameter n is the size of the board,the size of the board is n*n,the parameter k is the number of mines,'X' represents the mine,other numbers represent the number of mines around the position.
        :return: The minesweeper map, list.
        >>> minesweeper_game = MinesweeperGame(3, 1)
        >>> minesweeper_game.generate_mine_sweeper_map()
        [['X', 1, 0], [1, 1, 0], [0, 0, 0]]

        """
        mine_sweeper_map = [['-' for _ in range(self.n)] for _ in range(self.n)]
        placed_mines = 0
        while placed_mines < self.k:
            x = random.randint(0, self.n - 1)
            y = random.randint(0, self.n - 1)
            if mine_sweeper_map[x][y] != 'X':
                mine_sweeper_map[x][y] = 'X'
                placed_mines += 1
        for x in range(self.n):
            for y in range(self.n):
                if mine_sweeper_map[x][y] != 'X':
                    count = self.count_adjacent_mines(x, y, mine_sweeper_map)
                    mine_sweeper_map[x][y] = count
        return mine_sweeper_map

    def generate_playerMap(self):
        """
        Generates a player map with the given size of the board, the given parameter n is the size of the board,the size of the board is n*n,the parameter k is the number of mines,'-' represents the unknown position.
        :return: The player map, list.
        >>> minesweeper_game = MinesweeperGame(3, 1)
        >>> minesweeper_game.generate_playerMap()
        [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]

        """
        return [['-' for _ in range(self.n)] for _ in range(self.n)]

    def count_adjacent_mines(self, x, y, mine_sweeper_map):
        """
        Counts the number of mines adjacent to the given position.
        :param x: The x coordinate of the position, int.
        :param y: The y coordinate of the position, int.
        :param mine_sweeper_map: The minesweeper map, list.
        :return: The number of mines adjacent to the position, int.
        """
        count = 0
        for i in range(max(0, x - 1), min(self.n, x + 2)):
            for j in range(max(0, y - 1), min(self.n, y + 2)):
                if mine_sweeper_map[i][j] == 'X':
                    count += 1
        return count

    def check_won(self, map):
        """
        Checks whether the player has won the game,if there are just mines in the player map,return True,otherwise return False.
        :return: True if the player has won the game, False otherwise.
        >>> minesweeper_game = MinesweeperGame(3, 1)
        >>> minesweeper_game.minesweeper_map = [['X', 1, 0], [1, 1, 0], [0, 0, 0]]
        >>> minesweeper_game.player_map = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]
        >>> minesweeper_game.check_won(minesweeper_game.player_map)
        False

        """
        for row in map:
            for cell in row:
                if cell != 'X' and cell != '-':
                    return False
        return True

    def sweep(self, x, y):
        """
        Sweeps the given position.
        :param x: The x coordinate of the position, int.
        :param y: The y coordinate of the position, int.
        :return: True if the player has won the game, False otherwise,if the game still continues, return the player map, list.
        >>> minesweeper_game = MinesweeperGame(3, 1)
        >>> minesweeper_game.minesweeper_map = [['X', 1, 0], [1, 1, 0], [0, 0, 0]]
        >>> minesweeper_game.player_map = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]
        >>> minesweeper_game.sweep(1, 1)
        [['-', '-', '-'], ['-', 1, '-'], ['-', '-', '-']]

        """
        if self.player_map[x][y] != '-':
            return self.player_map
        if self.minesweeper_map[x][y] == 'X':
            self.player_map[x][y] = 'X'
            return True
        else:
            self.player_map[x][y] = self.minesweeper_map[x][y]
            return self.player_map



if __name__ == "__main__":
    instance = MinesweeperGame(3, 1)
    # Test case for generate_mine_sweeper_map
    output = instance.generate_mine_sweeper_map()
    print(f"Minesweeper Map: {output}")

    # Test case for generate_playerMap
    output = instance.generate_playerMap()
    print(f"Player Map: {output}")

    # Test case for count_adjacent_mines
    output = instance.count_adjacent_mines(1, 1, instance.minesweeper_map)
    print(f"Adjacent Mines: {output}")

    # Test case for check_won
    output = instance.check_won(instance.player_map)
    print(f"Has Won: {output}")

    # Test case for sweep
    output = instance.sweep(1, 1)
    print(f"Player Map after Sweep: {output}")