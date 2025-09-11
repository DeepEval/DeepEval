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
        board = [[0 for _ in range(self.n)] for _ in range(self.n)]
        mines = set()

        while len(mines) < self.k:
            x = random.randint(0, self.n - 1)
            y = random.randint(0, self.n - 1)
            if (x, y) not in mines:
                mines.add((x, y))
                board[x][y] = 'X'
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if 0 <= x + dx < self.n and 0 <= y + dy < self.n and board[x + dx][y + dy] != 'X':
                            board[x + dx][y + dy] += 1

        return board

    def generate_playerMap(self):
        """
        Generates a player map with the given size of the board.
        :return: The player map, list.
        """
        return [['-' for _ in range(self.n)] for _ in range(self.n)]

    def check_won(self, player_map):
        """
        Checks whether the player has won the game.
        :return: True if the player has won the game, False otherwise.
        """
        for row in range(self.n):
            for col in range(self.n):
                if player_map[row][col] == '-' and self.minesweeper_map[row][col] != 'X':
                    return False
        return True

    def sweep(self, x, y):
        """
        Sweeps the given position.
        :param x: The x coordinate of the position, int.
        :param y: The y coordinate of the position, int.
        :return: True if the player has won the game, False otherwise, if the game still continues, return the player map.
        """
        if self.minesweeper_map[x][y] == 'X':
            return "Game Over! You hit a mine."
        
        self.player_map[x][y] = self.minesweeper_map[x][y]
        
        if self.check_won(self.player_map):
            return True
        
        return self.player_map


if __name__ == "__main__":
    # Test the MinesweeperGame class
    minesweeper_game = MinesweeperGame(3, 1)

    # Test generate_mine_sweeper_map
    print("Minesweeper Map:")
    for row in minesweeper_game.minesweeper_map:
        print(row)

    # Test generate_playerMap
    print("\nPlayer Map:")
    for row in minesweeper_game.player_map:
        print(row)

    # Test check_won
    print("\nCheck if won (initially False):", minesweeper_game.check_won(minesweeper_game.player_map))

    # Test sweep
    print("\nSweep position (1, 1):")
    output = minesweeper_game.sweep(1, 1)
    print(output)

    # Check if won after the sweep
    print("\nCheck if won after sweep:", minesweeper_game.check_won(minesweeper_game.player_map))

    # Sweep more positions and check win condition
    print("\nSweeping more positions...")
    for i in range(minesweeper_game.n):
        for j in range(minesweeper_game.n):
            if minesweeper_game.player_map[i][j] == '-':
                result = minesweeper_game.sweep(i, j)
                print(f"Sweep ({i}, {j}):", result)
                if isinstance(result, bool) and result:
                    print("Player has won the game!")