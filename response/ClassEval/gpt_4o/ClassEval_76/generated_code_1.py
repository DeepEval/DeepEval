class SignInSystem:
    """
    This is a class for a sign-in system, including adding users, signing in/out, checking sign-in status, and retrieving signed-in/not signed-in users.
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
        if username in self.users:
            return False
        self.users[username] = False
        return True

    def sign_in(self, username):
        """
        Sign in a user if the user was in the self.users and change the state to True.
        :param username: str, the username to be signed in.
        :return: bool, True if the user is signed in successfully, False if the user does not exist.
        """
        if username in self.users:
            self.users[username] = True
            return True
        return False

    def check_sign_in(self, username):
        """
        Check if a user is signed in.
        :param username: str, the username to be checked.
        :return: bool, True if the user is signed in, False if the user does not exist or is not signed in.
        """
        return self.users.get(username, False)

    def all_signed_in(self):
        """
        Check if all users are signed in.
        :return: bool, True if all users are signed in, False otherwise.
        """
        return all(self.users.values())

    def all_not_signed_in(self):
        """
        Get a list of usernames that are not signed in.
        :return: list[str], a list of usernames that are not signed in.
        """
        return [username for username, signed_in in self.users.items() if not signed_in]


if __name__ == "__main__":
    # Test cases
    signInSystem = SignInSystem()

    # Test add_user method
    output = signInSystem.add_user("mike")
    print(output)  # True
    output = signInSystem.add_user("mike")
    print(output)  # False

    # Test sign_in method
    output = signInSystem.sign_in("mike")
    print(output)  # True
    output = signInSystem.sign_in("mik")
    print(output)  # False

    # Test check_sign_in method
    output = signInSystem.check_sign_in("mike")
    print(output)  # True
    output = signInSystem.check_sign_in("jack")
    print(output)  # False

    # Test all_signed_in method
    signInSystem.add_user("jack")
    signInSystem.sign_in("jack")
    output = signInSystem.all_signed_in()
    print(output)  # True

    # Test all_not_signed_in method
    signInSystem = SignInSystem()
    signInSystem.add_user("a")
    signInSystem.add_user("b")
    output = signInSystem.all_not_signed_in()
    print(output)  # ['a', 'b']