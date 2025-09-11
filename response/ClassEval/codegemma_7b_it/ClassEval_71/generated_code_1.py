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
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                if self.map[row][col] == 'O':
                    self.player_row = row
                    self.player_col = col
                elif self.map[row][col] == 'G':
                    self.targets.append((row, col))
                    self.target_count += 1
                elif self.map[row][col] == 'X':
                    self.boxes.append((row, col))

    def check_win(self):
        """
        Check if the game is won. The game is won when all the boxes are placed on target positions.
        And update the value of self.is_game_over.
        :return self.is_game_over: True if all the boxes are placed on target positions, or False otherwise.
        >>> game = PushBoxGame(["#####", "#O  #", "# X #", "#  G#", "#####"]) 
        >>> game.check_win()
        """
        self.is_game_over = True
        for box in self.boxes:
            if box not in self.targets:
                self.is_game_over = False
                break
        return self.is_game_over

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
            self.move_up()
        elif direction == 's':
            self.move_down()
        elif direction == 'a':
            self.move_left()
        elif direction == 'd':
            self.move_right()
        return self.check_win()

    def move_up(self):
        """
        Move the player up if the position is not a wall or a box on a wall.
        """
        if self.map[self.player_row - 1][self.player_col] == '#':
            return
        if self.map[self.player_row - 1][self.player_col] == 'X' or self.map[self.player_row - 1][self.player_col] == 'O':
            if self.map[self.player_row - 2][self.player_col] == '#':
                return
        self.map[self.player_row][self.player_col] = ' '
        self.player_row -= 1
        self.map[self.player_row][self.player_col] = 'O'

    def move_down(self):
        """
        Move the player down if the position is not a wall or a box on a wall.
        """
        if self.map[self.player_row + 1][self.player_col] == '#':
            return
        if self.map[self.player_row + 1][self.player_col] == 'X' or self.map[self.player_row + 1][self.player_col] == 'O':
            if self.map[self.player_row + 2][self.player_col] == '#':
                return
        self.map[self.player_row][self.player_col] = ' '
        self.player_row += 1
        self.map[self.player_row][self.player_col] = 'O'

    def move_left(self):
        """
        Move the player left if the position is not a wall or a box on a wall.
        """
        if self.map[self.player_row][self.player_col - 1] == '#':
            return
        if self.map[self.player_row][self.player_col - 1] == 'X' or self.map[self.player_row][self.player_col - 1] == 'O':
            if self.map[self.player_row][self.player_col - 2] == '#':
                return
        self.map[self.player_row][self.player_col] = ' '
        self.player_col -= 1
        self.map[self.player_row][self.player_col] = 'O'

    def move_right(self):
        """
        Move the player right if the position is not a wall or a box on a wall.
        """
        if self.map[self.player_row][self.player_col + 1] == '#':
            return
        if self.map[self.player_row][self.player_col + 1] == 'X' or self.map[self.player_row][self.player_col + 1] == 'O':
            if self.map[self.player_row][self.player_col + 2] == '#':
                return
        self.map[self.player_row][self.player_col] = ' '
        self.player_col += 1
        self.map[self.player_row][self.player_col] = 'O'

    def print_map(self):
        """
        Print the current state of the map.
        """
        for row in self.map:
            print(''.join(row))

if __name__ == "__main__":
    map = ["#####", "#O  #", "# X #", "#  G#", "#####"]
    game = PushBoxGame(map)
    game.print_map()
    print(game.check_win())
    print(game.move('d'))
    print(game.move('s'))
    print(game.move('a'))
    print(game.move('s'))
    print(game.move('d'))
    game.print_map()
    print(game.check_win())