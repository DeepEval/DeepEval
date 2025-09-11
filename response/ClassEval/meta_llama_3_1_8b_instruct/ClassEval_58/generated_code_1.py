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
        """
        minesweeper_map = [[0 for _ in range(self.n)] for _ in range(self.n)]
        mines = set()
        
        while len(mines) < self.k:
            x, y = random.randint(0, self.n - 1), random.randint(0, self.n - 1)
            if (x, y) not in mines:
                minesweeper_map[x][y] = 'X'
                mines.add((x, y))
        
        for i in range(self.n):
            for j in range(self.n):
                if minesweeper_map[i][j] == 'X':
                    continue
                count = 0
                for x in range(max(0, i-1), min(self.n, i+2)):
                    for y in range(max(0, j-1), min(self.n, j+2)):
                        if minesweeper_map[x][y] == 'X':
                            count += 1
                minesweeper_map[i][j] = count
        
        return minesweeper_map

    def generate_playerMap(self):
        """
        Generates a player map with the given size of the board, the given parameter n is the size of the board,the size of the board is n*n,the parameter k is the number of mines,'-' represents the unknown position.
        :return: The player map, list.
        """
        return [['-' for _ in range(self.n)] for _ in range(self.n)]

    def check_won(self, map):
        """
        Checks whether the player has won the game,if there are just mines in the player map,return True,otherwise return False.
        :return: True if the player has won the game, False otherwise.
        """
        for i in range(self.n):
            for j in range(self.n):
                if map[i][j]!= '-' and map[i][j]!= 'X':
                    return False
        return True

    def sweep(self, x, y):
        """
        Sweeps the given position.
        :param x: The x coordinate of the position, int.
        :param y: The y coordinate of the position, int.
        :return: True if the player has won the game, False otherwise,if the game still continues, return the player map, list.
        """
        if x < 0 or y < 0 or x >= self.n or y >= self.n:
            return "Invalid position"
        
        if self.player_map[x][y]!= '-':
            return "Position already revealed"
        
        if self.minesweeper_map[x][y] == 'X':
            self.player_map[x][y] = 'X'
            return True
        
        count = 0
        for i in range(max(0, x-1), min(self.n, x+2)):
            for j in range(max(0, y-1), min(self.n, y+2)):
                if self.player_map[i][j]!= '-':
                    continue
                if self.minesweeper_map[i][j]!= 'X':
                    count += 1
                self.player_map[i][j] = self.minesweeper_map[i][j]
        
        if count == 0:
            self.player_map[x][y] = self.minesweeper_map[x][y]
        
        return self.player_map

if __name__ == "__main__":
    minesweeper_game = MinesweeperGame(3, 1)
    
    # Test case for generate_mine_sweeper_map
    print(minesweeper_game.generate_mine_sweeper_map())
    
    # Test case for generate_playerMap
    print(minesweeper_game.generate_playerMap())
    
    # Test case for check_won
    minesweeper_game.minesweeper_map = [['X', 1, 0], [1, 1, 0], [0, 0, 0]]
    minesweeper_game.player_map = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]
    print(minesweeper_game.check_won(minesweeper_game.player_map))
    
    # Test case for sweep
    print(minesweeper_game.sweep(1, 1))