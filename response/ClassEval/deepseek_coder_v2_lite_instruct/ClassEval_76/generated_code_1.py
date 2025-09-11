class SignInSystem:
    """
    This is a class as sigin in system, including adding users, signing in/out, checking sign-in status, and retrieving signed-in/not signed-in users.
    """

    def __init__(self):
        """
        Initialize the sign-in system.
        """
        self.users = {}

    def add_user(self, username):
        """
        Add a user to the sign-in system if the user wasn't in the self.users.
        And the initial state is False.
        :param username: str, the username to be added.
        :return: bool, True if the user is added successfully, False if the user already exists.
        """
        if username not in self.users:
            self.users[username] = False
            return True
        else:
            return False

    def sign_in(self, username):
        """
        Sign in a user if the user was in the self.users and change the state to True.
        :param username: str, the username to be signed in.
        :return: bool, True if the user is signed in successfully, False if the user does not exist.
        """
        if username in self.users:
            self.users[username] = True
            return True
        else:
            return False

    def check_sign_in(self, username):
        """
        Check if a user is signed in.
        :param username: str, the username to be checked.
        :return: bool, True if the user is signed in, False if the user does not exist or is not signed in.
        """
        if username in self.users:
            return self.users[username]
        else:
            return False

    def all_signed_in(self):
        """
        Check if all users are signed in.
        :return: bool, True if all users are signed in, False otherwise.
        """
        return all(status for status in self.users.values())

    def all_not_signed_in(self):
        """
        Get a list of usernames that are not signed in.
        :return: list[str], a list of usernames that are not signed in.
        """
        return [username for username, status in self.users.items() if not status]


# Test cases
if __name__ == "__main__":
    signInSystem = SignInSystem()
    print(signInSystem.add_user("mike"))  # True
    print(signInSystem.add_user("mike"))  # False
    print(signInSystem.sign_in("mike"))   # True
    print(signInSystem.sign_in("mik"))    # False
    print(signInSystem.check_sign_in("jack"))  # False
    signInSystem.add_user("jack")
    print(signInSystem.check_sign_in("jack"))  # False
    signInSystem.sign_in("jack")
    print(signInSystem.check_sign_in("jack"))  # True
    print(signInSystem.all_signed_in())  # False
    signInSystem.add_user("jack")
    signInSystem.sign_in("jack")
    print(signInSystem.all_signed_in())  # True
    signInSystem = SignInSystem()
    signInSystem.add_user("a")
    signInSystem.add_user("b")
    print(signInSystem.all_not_signed_in())  # ['a', 'b']