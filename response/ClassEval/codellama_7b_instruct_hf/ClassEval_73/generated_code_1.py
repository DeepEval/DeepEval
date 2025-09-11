import math

class RPGCharacter:
    def __init__(self, name, hp, attack_power, defense, level=1):
        self.name = name
        self.hp = hp
        self.attack_power = attack_power
        self.defense = defense
        self.level = level
        self.exp = 0

    def attack(self, other_character):
        # Calculate the damage caused by the attack
        damage = self.attack_power - other_character.defense
        # Check if the damage is greater than 0
        if damage > 0:
            # Reduce the other character's hp by the damage
            other_character.hp -= damage
        # Return the other character's hp
        return other_character.hp

    def heal(self):
        # Check if the character's hp is less than the max hp
        if self.hp < self.max_hp:
            # Increase the character's hp by 10
            self.hp += 10
        # Return the character's hp
        return self.hp

    def gain_exp(self, amount):
        # Add the experience points to the character's exp
        self.exp += amount
        # Check if the character's exp is greater than the level requirement
        if self.exp > self.level * 100:
            # Increase the character's level by 1
            self.level += 1
            # Reset the character's exp to 0
            self.exp = 0
        # Return the character's exp
        return self.exp

    def level_up(self):
        # Increase the character's hp, attack power, and defense points by 5
        self.hp += 5
        self.attack_power += 5
        self.defense += 5
        # Return the character's new hp, attack power, and defense points
        return self.hp, self.attack_power, self.defense

    def is_alive(self):
        # Return True if the character's hp is greater than 0, or False otherwise
        return self.hp > 0

if __name__ == "__main__":

    player_1 = RPGCharacter("player 1", 100, 10, 3)
    player_1.max_hp = 100  # Add max_hp attribute
    player_2 = RPGCharacter("player 2", 100, 7, 2)
    player_2.max_hp = 100  # Add max_hp attribute

    output = player_1.attack(player_2)
    print(output)

    output = player_1.heal()
    print(output)

    player_1.gain_exp(100)
    output = player_1.gain_exp(100)
    print(output)

    output = player_1.level_up()
    print(output)

    output = player_1.is_alive()
    print(output)






