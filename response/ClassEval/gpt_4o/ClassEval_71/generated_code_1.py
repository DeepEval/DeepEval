class PushBoxGame:
    """
    This class implements a functionality of a sokoban game, where the player needs to move boxes to designated targets in order to win.
    """

    def __init__(self, map):
        """
        Initialize the push box game with the map and various attributes.
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
        """
        for row_idx, row in enumerate(self.map):
            for col_idx, char in enumerate(row):
                if char == 'O':
                    self.player_row = row_idx
                    self.player_col = col_idx
                elif char == 'G':
                    self.targets.append((row_idx, col_idx))
                elif char == 'X':
                    self.boxes.append((row_idx, col_idx))

    def check_win(self):
        """
        Check if the game is won. The game is won when all the boxes are placed on target positions.
        """
        for box in self.boxes:
            if box not in self.targets:
                self.is_game_over = False
                return self.is_game_over
        self.is_game_over = True
        return self.is_game_over

    def move(self, direction):
        """
        Move the player based on the specified direction and check if the game is won.
        """
        direction_map = {'w': (-1, 0), 's': (1, 0), 'a': (0, -1), 'd': (0, 1)}
        if direction in direction_map:
            dr, dc = direction_map[direction]
            new_row = self.player_row + dr
            new_col = self.player_col + dc
            
            if self.map[new_row][new_col] != '#':
                if (new_row, new_col) in self.boxes:
                    box_new_row = new_row + dr
                    box_new_col = new_col + dc
                    if self.map[box_new_row][box_new_col] != '#' and (box_new_row, box_new_col) not in self.boxes:
                        # Move the box
                        self.boxes.remove((new_row, new_col))
                        self.boxes.append((box_new_row, box_new_col))
                        # Move the player
                        self.player_row, self.player_col = new_row, new_col
                else:
                    # Move the player
                    self.player_row, self.player_col = new_row, new_col
        
        return self.check_win()

    def print_map(self):
        """
        Print the current map with the player's and boxes' current positions.
        """
        for row_idx, row in enumerate(self.map):
            row_list = list(row)
            if (row_idx, self.player_col) == (self.player_row, self.player_col):
                row_list[self.player_col] = 'O'
            for box in self.boxes:
                if box[0] == row_idx:
                    row_list[box[1]] = 'X'
            print(''.join(row_list))

if __name__ == "__main__":
    # Test case for init_game
    game = PushBoxGame(["#####", "#O  #", "# X #", "#  G#", "#####"]) 
    assert game.targets == [(3, 3)], "init_game failed: Incorrect targets"
    assert game.boxes == [(2, 2)], "init_game failed: Incorrect boxes"
    assert game.player_row == 1 and game.player_col == 1, "init_game failed: Incorrect player position"
    
    # Test case for check_win
    assert game.check_win() == False, "check_win failed: Game should not be won yet"
    
    # Test case for move
    game.print_map()
    assert game.move('d') == False, "move failed: Game should not be won yet"
    game.print_map()
    assert game.move('s') == False, "move failed: Game should not be won yet"
    game.print_map()
    assert game.move('a') == False, "move failed: Game should not be won yet"
    game.print_map()
    assert game.move('s') == False, "move failed: Game should not be won yet"
    game.print_map()
    assert game.move('d') == True, "move failed: Game should be won now"
    game.print_map()
    
    print("All test cases passed!")