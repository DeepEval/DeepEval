import numpy as np

class PushBoxGame:
    def __init__(self, map):
        self.map = map
        self.player_row = 0
        self.player_col = 0
        self.targets = []
        self.boxes = []
        self.target_count = 0
        self.is_game_over = False
        self.init_game()

    def init_game(self):
        self.map = np.array(self.map)
        self.player_row, self.player_col = np.where(self.map == 'O')
        self.targets = np.where(self.map == 'G')
        self.boxes = np.where(self.map == 'X')
        self.target_count = len(self.targets)
        self.is_game_over = False

    def check_win(self):
        if len(self.boxes) == 0:
            self.is_game_over = True
        return self.is_game_over

    def move(self, direction):
        if direction == 'w':
            self.player_row -= 1
        elif direction == 's':
            self.player_row += 1
        elif direction == 'a':
            self.player_col -= 1
        elif direction == 'd':
            self.player_col += 1

        if self.map[self.player_row, self.player_col] == 'X':
            self.boxes[self.player_row, self.player_col] = 'G'
            self.target_count -= 1
            if self.target_count == 0:
                self.is_game_over = True
        else:
            self.player_row, self.player_col = np.where(self.map == 'O')

        return self.is_game_over

if __name__ == "__main__":

    map = [
        list("#####"),
        list("#O  #"),
        list("# X #"),
        list("#  G#"),
        list("#####")
    ]
    game = PushBoxGame(map)
    moves = ['s', 'd', 'w']
    for direction in moves:
        print(game.map)
        print(f"Target count: {game.target_count}")
        print(f"Boxes: {game.boxes}")
        print(f"Player: {game.player_row}, {game.player_col}")
        print(f"Moving {direction}")
        game.move(direction)
        if game.is_game_over:
            break
    print("Game over")