import random

class MahjongConnect:
    def __init__(self, BOARD_SIZE, ICONS):
        self.BOARD_SIZE = BOARD_SIZE
        self.ICONS = ICONS
        self.board = self.create_board()

    def create_board(self):
        board = []
        for _ in range(self.BOARD_SIZE[0]):
            row = []
            for _ in range(self.BOARD_SIZE[1]):
                row.append(random.choice(self.ICONS))
            board.append(row)
        return board

    def is_valid_move(self, pos1, pos2):
        if pos1 == pos2:
            return False
        if not (0 <= pos1[0] < self.BOARD_SIZE[0] and 0 <= pos1[1] < self.BOARD_SIZE[1] and 0 <= pos2[0] < self.BOARD_SIZE[0] and 0 <= pos2[1] < self.BOARD_SIZE[1]):
            return False
        if self.board[pos1[0]][pos1[1]] != self.board[pos2[0]][pos2[1]]:
            return False
        if not self.has_path(pos1, pos2):
            return False
        return True

    def has_path(self, pos1, pos2):
        if pos1 == pos2:
            return True
        if self.board[pos1[0]][pos1[1]] == self.board[pos2[0]][pos2[1]]:
            return False
        if self.board[pos1[0]][pos1[1]] == ' ':
            return False
        if self.board[pos2[0]][pos2[1]] == ' ':
            return False
        return self.has_path((pos1[0], pos1[1]-1), pos2) or self.has_path((pos1[0], pos1[1]+1), pos2) or self.has_path((pos1[0]-1, pos1[1]), pos2) or self.has_path((pos1[0]+1, pos1[1]), pos2)

    def remove_icons(self, pos1, pos2):
        if self.is_valid_move(pos1, pos2):
            self.board[pos1[0]][pos1[1]] = ' '
            self.board[pos2[0]][pos2[1]] = ' '

    def is_game_over(self):
        for row in self.board:
            for icon in row:
                if icon != ' ':
                    return False
        return True

if __name__ == "__main__":
    mc = MahjongConnect([4, 4], ['a', 'b', 'c'])
    mc.board = [['a', 'b', 'c', 'a'],
                ['a', 'b', 'c', 'a'],
                ['a', 'b', 'c', 'a'],
                ['a', 'b', 'c', 'a']]
    print(mc.is_valid_move((0, 0), (1, 0))) # True
    print(mc.is_valid_move((0, 0), (0, 0))) # False
    print(mc.has_path((0, 0), (1, 0))) # True
    print(mc.has_path((0, 0), (0, 0))) # False
    mc.remove_icons((0, 0), (1, 0))
    print(mc.board) # [[' ', 'b', 'c', 'a'], [' ', 'b', 'c', 'a'], ['a', 'b', 'c', 'a'], ['a', 'b', 'c', 'a']]
    print(mc.is_game_over()) # False