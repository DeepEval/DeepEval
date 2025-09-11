class PushBoxGame:
    """
    This class implements a functionality of a sokoban game, where the player needs to move boxes to designated targets in order to win.
    """

    def __init__(self, game_map):
        """
        Initialize the push box game with the map and various attributes.
        :param game_map: list[str], the map of the push box game, represented as a list of strings. 
        """
        self.map = game_map
        self.player_row = 0
        self.player_col = 0
        self.targets = []
        self.boxes = []
        self.target_count = 0
        self.is_game_over = False
        self.init_game()

    def init_game(self):
        """
        Initialize the game by setting the positions of the player, targets, and boxes based on the map.
        """
        for r in range(len(self.map)):
            for c in range(len(self.map[r])):
                if self.map[r][c] == 'O':
                    self.player_row, self.player_col = r, c
                elif self.map[r][c] == 'G':
                    self.targets.append((r, c))
                    self.target_count += 1
                elif self.map[r][c] == 'X':
                    self.boxes.append((r, c))

    def check_win(self):
        """
        Check if the game is won. The game is won when all the boxes are placed on target positions.
        And update the value of self.is_game_over.
        :return: True if all the boxes are placed on target positions, or False otherwise.
        """
        self.is_game_over = all(box in self.targets for box in self.boxes)
        return self.is_game_over

    def move(self, direction):
        """
        Move the player based on the specified direction and check if the game is won.
        :param direction: str, the direction of the player's movement. 
        :return: True if the game is won, False otherwise.
        """
        row_change, col_change = 0, 0
        if direction == 'w':
            row_change = -1
        elif direction == 's':
            row_change = 1
        elif direction == 'a':
            col_change = -1
        elif direction == 'd':
            col_change = 1
        else:
            return False  # Invalid direction

        new_player_row = self.player_row + row_change
        new_player_col = self.player_col + col_change

        if self.map[new_player_row][new_player_col] == '#':
            return False  # Wall

        if (new_player_row, new_player_col) in self.boxes:
            new_box_row = new_player_row + row_change
            new_box_col = new_player_col + col_change
            if self.map[new_box_row][new_box_col] == '#' or (new_box_row, new_box_col) in self.boxes:
                return False  # Wall or another box

            # Move the box
            box_index = self.boxes.index((new_player_row, new_player_col))
            self.boxes[box_index] = (new_box_row, new_box_col)

        # Move the player
        self.player_row = new_player_row
        self.player_col = new_player_col
        
        return self.check_win()

    def print_map(self):
        """
        Print the current state of the map for visualization.
        """
        map_copy = [list(row) for row in self.map]
        map_copy[self.player_row][self.player_col] = 'O'
        for box in self.boxes:
            map_copy[box[0]][box[1]] = 'X'
        for target in self.targets:
            if target in self.boxes:
                map_copy[target[0]][target[1]] = 'G'
            else:
                map_copy[target[0]][target[1]] = 'G'
        
        for row in map_copy:
            print(''.join(row))

if __name__ == "__main__":
    # Test case for initialization
    game = PushBoxGame(["#####", "#O  #", "# X #", "#  G#", "#####"])
    print("Targets:", game.targets)  # Expected: [(3, 3)]
    print("Boxes:", game.boxes)      # Expected: [(2, 2)]
    print("Player position:", (game.player_row, game.player_col))  # Expected: (1, 1)

    # Test case for check_win
    print("Check win (before move):", game.check_win())  # Expected: False

    # Test case for moving
    game.print_map()
    print("Move right (d):", game.move('d'))  # Expected: False
    game.print_map()
    print("Move down (s):", game.move('s'))    # Expected: False
    game.print_map()
    print("Move left (a):", game.move('a'))     # Expected: False
    game.print_map()
    print("Move down (s):", game.move('s'))    # Expected: False
    game.print_map()
    print("Move right (d):", game.move('d'))  # Expected: True (if box reached target)
    game.print_map()