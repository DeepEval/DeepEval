class PushBoxGame:
    """
    This class implements a functionality of a sokoban game, where the player needs to move boxes to designated targets in order to win.
    """

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
        for row_index, row in enumerate(self.map):
            for col_index, char in enumerate(row):
                if char == 'O':
                    self.player_row = row_index
                    self.player_col = col_index
                elif char == 'G':
                    self.targets.append((row_index, col_index))
                    self.target_count += 1
                elif char == 'X':
                    self.boxes.append((row_index, col_index))

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
        self.is_game_over = True
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
            new_row = self.player_row - 1
            new_col = self.player_col
        elif direction == 's':
            new_row = self.player_row + 1
            new_col = self.player_col
        elif direction == 'a':
            new_row = self.player_row
            new_col = self.player_col - 1
        elif direction == 'd':
            new_row = self.player_row
            new_col = self.player_col + 1
        else:
            return False

        if self.map[new_row][new_col] != '#':
            self.player_row = new_row
            self.player_col = new_col
            self.move_box(direction)
            return self.check_win()
        return False

    def move_box(self, direction):
        """
        Move a box in the direction of the player.
        """
        for box_index, box_pos in enumerate(self.boxes):
            if box_pos[0] == self.player_row and box_pos[1] == self.player_col:
                if direction == 'w':
                    new_box_row = box_pos[0] - 1
                    new_box_col = box_pos[1]
                elif direction == 's':
                    new_box_row = box_pos[0] + 1
                    new_box_col = box_pos[1]
                elif direction == 'a':
                    new_box_row = box_pos[0]
                    new_box_col = box_pos[1] - 1
                elif direction == 'd':
                    new_box_row = box_pos[0]
                    new_box_col = box_pos[1] + 1
                else:
                    return
                if self.map[new_box_row][new_box_col] != '#':
                    self.boxes[box_index] = (new_box_row, new_box_col)



    def print_map(self):
        """
        Print the current state of the game map.
        >>> game = PushBoxGame(["#####", "#O  #", "# X #", "#  G#", "#####"])       
        >>> game.print_map()
        # # # # # 
        # O     #
        #   X   #
        #     G #
        # # # # #
        """
        for row in self.map:
            print(''.join(row))



if __name__ == "__main__":
    map = ["#####", "#O  #", "# X #", "#  G#", "#####"]   
    game = PushBoxGame(map)
    game.print_map()
    print(game.move('d'))
    game.print_map()