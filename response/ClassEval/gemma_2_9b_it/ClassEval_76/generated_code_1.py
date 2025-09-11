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
        >>> signInSystem.add_user("mike")
        True
        >>> signInSystem.add_user("mike")
        False
        """
        if username not in self.users:
            self.users[username] = False
            return True
        return False

    def sign_in(self, username):
        """
        Sign in a user if the user was in the self.users and change the state to True.
        :param username: str, the username to be signed in.
        :return: bool, True if the user is signed in successfully, False if the user does not exist.
        >>> signInSystem.sign_in("mike")
        True
        >>> signInSystem.sign_in("mik")
        False
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
        >>> signInSystem.check_sign_in("jack")
        False
        >>> signInSystem.add_user("jack")
        >>> signInSystem.check_sign_in("jack")
        >>> signInSystem.sign_in("jack")
        >>> signInSystem.check_sign_in("jack")
        True
        """
        if username in self.users and self.users[username]:
            return True
        return False

    def all_signed_in(self):
        """
        Check if all users are signed in.
        :return: bool, True if all users are signed in, False otherwise.
        >>> signInSystem.add_user("jack")
        True
        >>> signInSystem.sign_in("jack")
        >>> signInSystem.all_signed_in()
        True
        """
        for user in self.users.values():
            if not user:
                return False
        return True

    def all_not_signed_in(self):
        """
        Get a list of usernames that are not signed in.
        :return: list[str], a list of usernames that are not signed in.
        >>> signInSystem = SignInSystem()
        >>> signInSystem.add_user("a")
        True
        >>> signInSystem.add_user("b")
        True
        >>> signInSystem.all_not_signed_in()
        ['a', 'b']
        """
        return [username for username, signed_in in self.users.items() if not signed_in]


if __name__ == "__main__":
    signInSystem = SignInSystem()
    # Test case for add_user
    print(signInSystem.add_user("mike"))  # Output: True
    print(signInSystem.add_user("mike"))  # Output: False

    # Test case for sign_in
    print(signInSystem.sign_in("mike"))  # Output: True
    print(signInSystem.sign_in("mik"))  # Output: False

    # Test case for check_sign_in
    print(signInSystem.check_sign_in("jack"))  # Output: False
    signInSystem.add_user("jack")
    print(signInSystem.check_sign_in("jack"))  # Output: False
    signInSystem.sign_in("jack")
    print(signInSystem.check_sign_in("jack"))  # Output: True

    # Test case for all_signed_in
    print(signInSystem.all_signed_in())  # Output: False
    signInSystem.sign_in("jack")
    print(signInSystem.all_signed_in())  # Output: True

    # Test case for all_not_signed_in
    print(signInSystem.all_not_signed_in())  # Output: ['mike']