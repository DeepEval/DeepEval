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
        Generates a minesweeper map with the given size of the board and the number of mines,
        the given parameter n is the size of the board, the size of the board is n*n,
        the parameter k is the number of mines, 'X' represents the mine, other numbers represent the number of mines around the position.
        :return: The minesweeper map, list.
        """
        # Initialize the map with zeros
        map = [[0 for _ in range(self.n)] for _ in range(self.n)]
        
        # Place mines randomly
        mines_placed = 0
        while mines_placed < self.k:
            x = random.randint(0, self.n - 1)
            y = random.randint(0, self.n - 1)
            if map[x][y] != 'X':
                map[x][y] = 'X'
                mines_placed += 1
        
        # Update the map with the number of adjacent mines
        for i in range(self.n):
            for j in range(self.n):
                if map[i][j] == 'X':
                    continue
                count = 0
                for x in range(max(0, i-1), min(self.n, i+2)):
                    for y in range(max(0, j-1), min(self.n, j+2)):
                        if map[x][y] == 'X':
                            count += 1
                map[i][j] = count
        
        return map

    def generate_playerMap(self):
        """
        Generates a player map with the given size of the board, the given parameter n is the size of the board,
        the size of the board is n*n, the parameter k is the number of mines, '-' represents the unknown position.
        :return: The player map, list.
        """
        return [['-' for _ in range(self.n)] for _ in range(self.n)]

    def check_won(self, map):
        """
        Checks whether the player has won the game, if there are just mines in the player map, return True, otherwise return False.
        :return: True if the player has won the game, False otherwise.
        """
        for i in range(self.n):
            for j in range(self.n):
                if self.minesweeper_map[i][j] != 'X' and map[i][j] != self.minesweeper_map[i][j]:
                    return False
        return True

    def sweep(self, x, y):
        """
        Sweeps the given position.
        :param x: The x coordinate of the position, int.
        :param y: The y coordinate of the position, int.
        :return: True if the player has won the game, False otherwise,
        if the game still continues, return the player map, list.
        """
        if self.minesweeper_map[x][y] == 'X':
            return True
        else:
            self.player_map[x][y] = self.minesweeper_map[x][y]
            return self.check_won(self.player_map)

if __name__ == "__main__":
    # Test cases
    minesweeper_game = MinesweeperGame(3, 1)
    print("Minesweeper Map:")
    print(minesweeper_game.minesweeper_map)
    print("Player Map:")
    print(minesweeper_game.player_map)
    print("Sweeping (1, 1):")
    won = minesweeper_game.sweep(1, 1)
    print(minesweeper_game.player_map)
    if won:
        print("You won the game!")
    else:
        print("Game still continues.")