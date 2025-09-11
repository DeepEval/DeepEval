class RPGCharacter:
    def __init__(self, name, hp, attack_power, defense, level=1):
        self.name = name
        self.hp = hp
        self.attack_power = attack_power
        self.defense = defense
        self.level = level
        self.exp = 0

    def attack(self, other_character):
        damage = max(self.attack_power - other_character.defense, 0)
        other_character.hp -= damage

    def heal(self):
        self.hp = min(100, self.hp + 10)

    def gain_exp(self, amount):
        self.exp += amount
        if self.exp >= 100 * self.level:
            self.exp -= 100 * self.level
            self.level_up()

    def level_up(self):
        hp_increase = 20
        attack_power_increase = 5
        defense_increase = 5

        self.hp += hp_increase
        self.attack_power += attack_power_increase
        self.defense += defense_increase
        self.level += 1
        self.exp = 0
        return self.level, self.hp, self.attack_power, self.defense

    def is_alive(self):
        return self.hp > 0

# Test cases
if __name__ == "__main__":
    player_1 = RPGCharacter('player 1', 100, 10, 3)
    player_2 = RPGCharacter('player 2', 100, 7, 2)
    print(player_1.is_alive())  # Expected: True
    player_1.attack(player_2)
    print(player_2.hp)  # Expected: 92
    player_1.heal()
    print(player_1.hp)  # Expected: 100
    player_1.gain_exp(1100)
    print(player_1.exp)  # Expected: 100
    print(player_1.level)  # Expected: 5
    new_level, new_hp, new_attack, new_defense = player_1.level_up()
    print(new_level, new_hp, new_attack, new_defense)  # Expected: (2, 120, 15, 8)
    print(player_1.is_alive())  # Expected: True