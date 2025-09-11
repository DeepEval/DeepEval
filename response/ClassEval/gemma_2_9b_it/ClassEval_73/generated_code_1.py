import random

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
        >>> player_1 = RPGCharacter('player 1', 100, 10, 3)
        >>> player_2 = RPGCharacter('player 2', 100, 7, 2)
        >>> player_1.attack(player_2)
        >>> player_2.hp
        92
        """
        damage = max(0, self.attack_power - other_character.defense)
        other_character.hp -= damage
        print(f"{self.name} attacks {other_character.name} for {damage} damage!")

    def heal(self):
        """
        Heal the character with 10 hp and the max hp is 100.
        :return: int, the current health points after healing.
        >>> player_1 = RPGCharacter('player 1', 93, 10, 3)
        >>> player_1.heal()
        100
        """
        self.hp = min(self.hp + 10, 100)
        print(f"{self.name} heals for 10 HP.")
        return self.hp

    def gain_exp(self, amount):
        """
        Gain experience points for the character and level_up when the exp has reached the values that is 100 times the current level.
        The experience that overflows should be used to calculate the next level up until exhausts.
        :param amount: int, the amount of experience points to gain.
        >>> player_1 = RPGCharacter('player 1', 100, 10, 3)
        >>> player_1.gain_exp(1100)
        >>> player_1.exp
        100
        >>> player_1.level
        5
        """
        self.exp += amount
        while self.exp >= self.level * 100:
            self.level_up()

    def level_up(self):
        """
        Level up the character and return to zero experience points, increase hp by 20 points, attack power and defense points by 5 points.
        Max level is 100.
        :return: tuple[int, int, int, int], the new level, health points, attack power, and defense points after leveling up.
        >>> player_1 = RPGCharacter('player 1', 100, 10, 3)
        >>> player_1.level_up()
        (2, 120, 15, 8)
        """
        self.exp = 0
        self.level = min(self.level + 1, 100)
        self.hp += 20
        self.attack_power += 5
        self.defense += 5
        print(f"{self.name} leveled up to level {self.level}!")
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
    player_1 = RPGCharacter('player 1', 100, 10, 3)

    # Test case for attack
    player_2 = RPGCharacter('player 2', 100, 7, 2)
    player_1.attack(player_2)
    print(f"{player_2.name}'s HP: {player_2.hp}")

    # Test case for heal
    player_1.heal()
    print(f"{player_1.name}'s HP: {player_1.hp}")

    # Test case for gain_exp
    player_1.gain_exp(1100)
    print(f"{player_1.name}'s level: {player_1.level}")
    print(f"{player_1.name}'s exp: {player_1.exp}")

    # Test case for level_up
    player_1.level_up()
    print(f"{player_1.name}'s level: {player_1.level}")
    print(f"{player_1.name}'s HP: {player_1.hp}")
    print(f"{player_1.name}'s attack power: {player_1.attack_power}")
    print(f"{player_1.name}'s defense: {player_1.defense}")

    # Test case for is_alive
    print(f"{player_1.name} is alive: {player_1.is_alive()}")