import unittest

class RPGCharacter:
    """
    The class represents a role-playing game character, which allows to attack other characters, heal, gain experience, level up, and check if the character is alive.
    """

    def __init__(self, name, hp, attack_power, defense, level=1):
        """
        Initialize an RPG character object.
        :param name: str, the name of the character.
        :param hp: int, The health points of the character.
        :param attack_power: int, the attack power of the character.
        :param defense: int, the defense points of the character.
        :param level: int, the level of the character. Default is 1.
        """
        self.name = name
        self.hp = hp
        self.attack_power = attack_power
        self.defense = defense
        self.level = level
        self.exp = 0

    def attack(self, other_character):
        """
        Attack another character. The damage caused needs to offset the defense value.
        :param other_character: RPGCharacter, The character being attacked.
        :return: int, the damage caused to the other character.
        >>> player_1 = RPGCharacter('player 1', 100, 10, 3)
        >>> player_2 = RPGCharacter('player 2', 100, 7, 2)
        >>> player_1.attack(player_2)
        6
        """
        damage = max(0, self.attack_power - other_character.defense)
        other_character.hp -= damage
        return damage

    def heal(self):
        """
        Heal the character with 10 hp and the max hp is 100.
        :return: int, the current health points after healing.
        >>> player_1 = RPGCharacter('player 1', 93, 10, 3)
        >>> player_1.heal()
        100
        """
        if self.hp < 100:
            self.hp = min(100, self.hp + 10)
        return self.hp

    def gain_exp(self, amount):
        """
        Gain experience points for the character and level_up when the exp has reached the values that is 100 times the current level
        The experience that overflows should be used to calculate the next leve up untill exhausts
        :param amount: int, the amount of experience points to gain.
        >>> player_1 = RPGCharacter('player 1', 100, 10, 3)
        >>> player_1.gain_exp(1100)
        >>> player_1.exp
        100
        >>> player_1.level
        5
        """
        self.exp += amount
        while self.exp >= 100 * self.level:
            self.level_up()
            self.exp -= 100 * self.level

    def level_up(self):
        """
        Level up the character and return to zero experience points, increase hp by 20 points, attack power and defense points by 5 points.
        max level is 100
        :return: tuple[int, int, int, int], the new level, health points, attack power, and defense points after leveling up.
        >>> player_1 = RPGCharacter('player 1', 100, 10, 3)
        >>> player_1.level_up()
        (2, 120, 15, 8)
        """
        if self.level < 100:
            self.level += 1
            self.hp = min(100, self.hp + 20)
            self.attack_power += 5
            self.defense += 5
            self.exp = 0
        return self.level, self.hp, self.attack_power, self.defense

    def is_alive(self):
        """
        Check if player is alive.
        :return: True if the hp is larger than 0, or False otherwise.
        >>> player_1 = RPGCharacter('player 1', 100, 10, 3)
        >>> player_1.is_alive()
        True
        """
        return self.hp > 0

if __name__ == "__main__":
    # Test case for attack method
    player_1 = RPGCharacter('player 1', 100, 10, 3)
    player_2 = RPGCharacter('player 2', 100, 7, 2)
    print(player_1.attack(player_2))  # Output: 6

    # Test case for heal method
    player_1 = RPGCharacter('player 1', 93, 10, 3)
    print(player_1.heal())  # Output: 100

    # Test case for gain_exp method
    player_1 = RPGCharacter('player 1', 100, 10, 3)
    player_1.gain_exp(1100)
    print(player_1.exp)  # Output: 100
    print(player_1.level)  # Output: 5

    # Test case for level_up method
    player_1 = RPGCharacter('player 1', 100, 10, 3)
    print(player_1.level_up())  # Output: (2, 120, 15, 8)

    # Test case for is_alive method
    player_1 = RPGCharacter('player 1', 100, 10, 3)
    print(player_1.is_alive())  # Output: True

class TestRPGCharacter(unittest.TestCase):

    def test_attack(self):
        player_1 = RPGCharacter('player 1', 100, 10, 3)
        player_2 = RPGCharacter('player 2', 100, 7, 2)
        self.assertEqual(player_1.attack(player_2), 6)

    def test_heal(self):
        player_1 = RPGCharacter('player 1', 93, 10, 3)
        self.assertEqual(player_1.heal(), 100)

    def test_gain_exp(self):
        player_1 = RPGCharacter('player 1', 100, 10, 3)
        player_1.gain_exp(1100)
        self.assertEqual(player_1.exp, 100)
        self.assertEqual(player_1.level, 5)

    def test_level_up(self):
        player_1 = RPGCharacter('player 1', 100, 10, 3)
        level, hp, attack_power, defense = player_1.level_up()
        self.assertEqual(level, 2)
        self.assertEqual(hp, 120)
        self.assertEqual(attack_power, 15)
        self.assertEqual(defense, 8)

    def test_is_alive(self):
        player_1 = RPGCharacter('player 1', 100, 10, 3)
        self.assertTrue(player_1.is_alive())

if __name__ == "__main__":
    unittest.main()