class PushBoxGame:
    def __init__(self, map):
        """
        Initialize the push box game with the map and various attributes.
        :param map: list[str], the map of the push box game, represented as a list of strings. 
            Each character on the map represents a different element, including the following:
            - '#' represents a wall that neither the player nor the box can pass through;
            - 'O' represents the initial position of the player;
            - 'G' represents the target position;
            - 'X' represents the initial position of the box.
        >>> map = ["#####", "#O  #", "# X #", "#  G#", "#####"]   
        >>> game = PushBoxGame(map)                
        """
        self.map = map
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
        >>> game = PushBoxGame(["#####", "#O  #", "# X #", "#  G#", "#####"]) 
        >>> game.targets
        [(3, 3)]
        >>> game.boxes
        [(2, 2)]
        >>> game.player_row
        1
        >>> game.player_col
        1
        """
        for row, line in enumerate(self.map):
            for col, cell in enumerate(line):
                if cell == 'O':
                    self.player_row, self.player_col = row, col
                elif cell == 'G':
                    self.targets.append((row, col))
                elif cell == 'X':
                    self.boxes.append((row, col))

    def check_win(self):
        """
        Check if the game is won. The game is won when all the boxes are placed on target positions.
        And update the value of self.is_game_over.
        :return self.is_game_over: True if all the boxes are placed on target positions, or False otherwise.
        >>> game = PushBoxGame(["#####", "#O  #", "# X #", "#  G#", "#####"]) 
        >>> game.check_win()
        """
        for box_pos in self.boxes:
            if box_pos not in self.targets:
                return False
        return True

    def move(self, direction):
        """
        Move the player based on the specified direction and check if the game is won.
        :param direction: str, the direction of the player's movement. 
            It can be 'w', 's', 'a', or 'd' representing up, down, left, or right respectively.

        :return: True if the game is won, False otherwise.
        >>> game = PushBoxGame(["#####", "#O  #", "# X #", "#  G#", "#####"])       
        >>> game.print_map()
        # # # # # 
        # O     #
        #   X   #
        #     G #
        # # # # #
        >>> game.move('d')
        False
        >>> game.move('s')   
        False
        >>> game.move('a')   
        False
        >>> game.move('s') 
        False
        >>> game.move('d') 
        True
        """
        if direction == 'w':
            self.player_row -= 1
        elif direction == 's':
            self.player_row += 1
        elif direction == 'a':
            self.player_col -= 1
        elif direction == 'd':
            self.player_col += 1

        # Check if the player is off the map
        if self.player_row < 0 or self.player_row >= len(self.map) or \
           self.player_col < 0 or self.player_col >= len(self.map[self.player_row]):
            return False

        # Check if the player hits a wall or a box
        if self.map[self.player_row][self.player_col] in ('#', 'X'):
            self.player_row, self.player_col = self.move_back()
            return False

        # Check if the player reaches a target
        for target_row, target_col in self.targets:
            if self.player_row == target_row and self.player_col == target_col:
                self.targets.remove((target_row, target_col))
                self.move(direction)  # Move the box to the target
                self.player_row, self.player_col = self.move_back()
                return self.check_win()

        return False

    def move_back(self):
        """
        Move the player back to the previous cell if they moved off the map.
        """
        if self.player_row > 0:
            self.player_row -= 1
        elif self.player_row < len(self.map) - 1:
            self.player_row += 1
        elif self.player_col > 0:
            self.player_col -= 1
        elif self.player_col < len(self.map[self.player_row]) - 1:
            self.player_col += 1
        return self.player_row, self.player_col

    def print_map(self):
        """
        Print the current state of the map.
        """
        for row in self.map:
            print(''.join(row))

# Test cases
if __name__ == "__main__":
    game = PushBoxGame(["#####", "#O  #", "# X #", "#  G#", "#####"])
    print("Initial Map:")
    game.print_map()

    print("Moving 'd' and checking win...")
    if game.move('d'):
        print("Win!")
    else:
        print("Lose.")

    print("Moving 's' and checking win...")
    if game.move('s'):
        print("Win!")
    else:
        print("Lose.")

    print("Moving 'a' and checking win...")
    if game.move('a'):
        print("Win!")
    else:
        print("Lose.")

    print("Moving 's' again and checking win...")
    if game.move('s'):
        print("Win!")
    else:
        print("Lose.")

    print("Moving 'd' again and checking win...")
    if game.move('d'):
        print("Win!")
    else:
        print("Lose.")