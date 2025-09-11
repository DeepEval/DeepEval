import random

class MinesweeperGame:
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.minesweeper_map = self.generate_mine_sweeper_map()
        self.player_map = self.generate_playerMap()
        self.score = 0

    def generate_mine_sweeper_map(self):
        minesweeper_map = []
        for i in range(self.n):
            row = []
            for j in range(self.n):
                if random.randint(0, 1) == 0:
                    row.append('X')
                else:
                    row.append(0)
            minesweeper_map.append(row)
        return minesweeper_map

    def generate_playerMap(self):
        player_map = []
        for i in range(self.n):
            row = []
            for j in range(self.n):
                row.append('-')
            player_map.append(row)
        return player_map

    def check_won(self, map):
        for row in map:
            for cell in row:
                if cell == 'X':
                    return False
        return True

    def sweep(self, x, y):
        if self.minesweeper_map[x][y] == 'X':
            return False
        self.player_map[x][y] = 'O'
        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                if 0 <= i < self.n and 0 <= j < self.n:
                    if self.minesweeper_map[i][j] == 'X':
                        self.player_map[i][j] = 'X'
        return self.player_map

if __name__ == "__main__":
    minesweeper_game = MinesweeperGame(3, 1)
    minesweeper_game.minesweeper_map = [['X', 1, 0], [1, 1, 0], [0, 0, 0]]
    minesweeper_game.player_map = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]
    print(minesweeper_game.sweep(1, 1))
    print(minesweeper_game.check_won(minesweeper_game.player_map))