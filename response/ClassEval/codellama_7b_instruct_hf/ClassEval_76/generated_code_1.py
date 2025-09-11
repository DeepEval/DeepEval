import collections

class SignInSystem:
    def __init__(self):
        self.users = collections.defaultdict(lambda: False)

    def add_user(self, username):
        if username in self.users:
            return False
        self.users[username] = False
        return True

    def sign_in(self, username):
        if username not in self.users:
            return False
        self.users[username] = True
        return True

    def check_sign_in(self, username):
        if username not in self.users:
            return False
        return self.users[username]

    def all_signed_in(self):
        for user in self.users:
            if not self.users[user]:
                return False
        return True

    def all_not_signed_in(self):
        not_signed_in = []
        for user in self.users:
            if not self.users[user]:
                not_signed_in.append(user)
        return not_signed_in
    
if __name__ == "__main__":
    system = SignInSystem()
    print(system.add_user("Alice"))  # True
    print(system.add_user("Bob"))    # True
    print(system.add_user("Alice"))  # False
    print(system.sign_in("Alice"))   # True
    print(system.sign_in("Charlie"))  # False
    print(system.check_sign_in("Alice"))  # True
    print(system.check_sign_in("Bob"))    # False
    print(system.all_signed_in())          # False
    print(system.all_not_signed_in())      # ['Bob']