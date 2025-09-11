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
        Generates a minesweeper map with the given size of the board and the number of mines.
        :return: The minesweeper map, list.
        """
        # Create an empty board
        board = [[0] * self.n for _ in range(self.n)]

        # Place mines randomly
        mines = set()
        while len(mines) < self.k:
            x, y = random.randint(0, self.n - 1), random.randint(0, self.n - 1)
            if (x, y) not in mines:
                mines.add((x, y))
                board[x][y] = 'X'

                # Update numbers around the mine
                for i in range(max(0, x - 1), min(self.n, x + 2)):
                    for j in range(max(0, y - 1), min(self.n, y + 2)):
                        if board[i][j] != 'X':
                            board[i][j] += 1
        return board

    def generate_playerMap(self):
        """
        Generates a player map with the given size of the board.
        :return: The player map, list.
        """
        return [['-'] * self.n for _ in range(self.n)]

    def check_won(self, player_map):
        """
        Checks whether the player has won the game.
        :return: True if the player has won the game, False otherwise.
        """
        for i in range(self.n):
            for j in range(self.n):
                if player_map[i][j] == '-' and self.minesweeper_map[i][j] != 'X':
                    return False
        return True

    def sweep(self, x, y):
        """
        Sweeps the given position.
        :param x: The x coordinate of the position, int.
        :param y: The y coordinate of the position, int.
        :return: True if the player has won the game, False otherwise, or the player map.
        """
        if self.minesweeper_map[x][y] == 'X':
            self.player_map[x][y] = 'X'
            return False  # The game is over as a mine is hit

        self.player_map[x][y] = self.minesweeper_map[x][y]
        if self.check_won(self.player_map):
            return True

        return self.player_map


if __name__ == "__main__":
    # Test case for MinesweeperGame
    minesweeper_game = MinesweeperGame(3, 1)

    # Test generate_mine_sweeper_map
    print("Minesweeper Map:")
    print(minesweeper_game.generate_mine_sweeper_map())

    # Test generate_playerMap
    print("Player Map:")
    print(minesweeper_game.generate_playerMap())

    # Test check_won
    minesweeper_game.minesweeper_map = [['X', 1, 0], [1, 1, 0], [0, 0, 0]]
    minesweeper_game.player_map = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]
    print("Check Won:", minesweeper_game.check_won(minesweeper_game.player_map))

    # Test sweep
    print("Sweep Result:")
    print(minesweeper_game.sweep(1, 1))