class PushBoxGame:
    """
    This class implements a functionality of a sokoban game, where the player needs
    to move boxes to designated targets in order to win.
    """

    def __init__(self, map):
        """
        Initialize the push box game with the map and various attributes.
        :param map: list[str], the map of the push box game, represented as a list
        of strings. 
            Each character on the map represents a different element, including
            the following:
            - '#' represents a wall that neither the player nor the box can pass
            through;
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
        Initialize the game by setting the positions of the player, targets, and
        boxes based on the map.
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
        for i, row in enumerate(self.map):
            for j, char in enumerate(row):
                if char == 'O':
                    self.player_row = i
                    self.player_col = j
                elif char == 'G':
                    self.targets.append((i, j))
                    self.target_count += 1
                elif char == 'X':
                    self.boxes.append((i, j))

    def check_win(self):
        """
        Check if the game is won. The game is won when all the boxes are placed
        on target positions.
        And update the value of self.is_game_over.
        :return self.is_game_over: True if all the boxes are placed on target positions,
        or False otherwise.
        >>> game = PushBoxGame(["#####", "#O  #", "# X #", "#  G#", "#####"]) 
        >>> game.check_win()
        """
        for box in self.boxes:
            if box not in self.targets:
                return False
        self.is_game_over = True
        return True

    def move(self, direction):
        """
        Move the player based on the specified direction and check if the game
        is won.
        :param direction: str, the direction of the player's movement. 
            It can be 'w','s', 'a', or 'd' representing up, down, left, or right respectively.

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
        new_row = self.player_row
        new_col = self.player_col

        if direction == 'w' and self.player_row > 0 and self.map[self.player_row - 1][self.player_col]!= '#':
            new_row -= 1
        elif direction =='s' and self.player_row < len(self.map) - 1 and self.map[self.player_row + 1][self.player_col]!= '#':
            new_row += 1
        elif direction == 'a' and self.player_col > 0 and self.map[self.player_row][self.player_col - 1]!= '#':
            new_col -= 1
        elif direction == 'd' and self.player_col < len(self.map[0]) - 1 and self.map[self.player_row][self.player_col + 1]!= '#':
            new_col += 1

        if self.map[new_row][new_col] == 'X':
            self.map[self.player_row][self.player_col] = 'O'
            self.player_row = new_row
            self.player_col = new_col
            self.map[self.player_row][self.player_col] = 'O'
            self.boxes.remove((self.player_row, self.player_col))
            self.boxes.append((new_row, new_col))
        elif self.map[new_row][new_col] == 'G':
            self.map[self.player_row][self.player_col] = 'O'
            self.player_row = new_row
            self.player_col = new_col
            self.map[self.player_row][self.player_col] = 'O'
            self.boxes.remove((self.player_row, self.player_col))
            self.targets.remove((new_row, new_col))
            self.target_count -= 1
        elif self.map[new_row][new_col] =='':
            self.map[self.player_row][self.player_col] = 'O'
            self.player_row = new_row
            self.player_col = new_col
            self.map[self.player_row][self.player_col] = 'O'

        self.check_win()
        return self.is_game_over

    def print_map(self):
        for row in self.map:
            print(row)

if __name__ == "__main__":
    map = ["#####", "#O  #", "# X #", "#  G#", "#####"]   
    game = PushBoxGame(map)
    print("Initial Map:")
    game.print_map()
    print("\nMove 'd':")
    game.move('d')
    print("\nFinal Map:")
    game.print_map()
    print("\nIs Game Over?", game.is_game_over)
    print("Targets:", game.targets)
    print("Boxes:", game.boxes)