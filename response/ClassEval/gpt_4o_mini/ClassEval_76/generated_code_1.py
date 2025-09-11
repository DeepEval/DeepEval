class SignInSystem:
    """
    This is a class as sign-in system, including adding users, signing in/out, 
    checking sign-in status, and retrieving signed-in/not signed-in users.
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
        self.users[username] = False  # Initial state is False (not signed in)
        return True

    def sign_in(self, username):
        """
        Sign in a user if the user was in the self.users and change the state to True.
        :param username: str, the username to be signed in.
        :return: bool, True if the user is signed in successfully, False if the user does not exist.
        """
        if username not in self.users:
            return False
        self.users[username] = True  # Change state to True (signed in)
        return True

    def check_sign_in(self, username):
        """
        Check if a user is signed in.
        :param username: str, the username to be checked.
        :return: bool, True if the user is signed in, False if the user does not exist or is not signed in.
        """
        if username not in self.users:
            return False
        return self.users[username]  # Return sign-in status

    def all_signed_in(self):
        """
        Check if all users are signed in.
        :return: bool, True if all users are signed in, False otherwise.
        """
        return all(self.users.values())  # All users should have True state

    def all_not_signed_in(self):
        """
        Get a list of usernames that are not signed in.
        :return: list[str], a list of usernames that are not signed in.
        """
        return [username for username, signed_in in self.users.items() if not signed_in]


if __name__ == "__main__":
    signInSystem = SignInSystem()
    
    # Test add_user
    print(signInSystem.add_user("mike"))  # True
    print(signInSystem.add_user("mike"))  # False

    # Test sign_in
    print(signInSystem.sign_in("mike"))  # True
    print(signInSystem.sign_in("mik"))   # False

    # Test check_sign_in
    print(signInSystem.check_sign_in("jack"))  # False
    signInSystem.add_user("jack")
    print(signInSystem.check_sign_in("jack"))  # False
    signInSystem.sign_in("jack")
    print(signInSystem.check_sign_in("jack"))  # True

    # Test all_signed_in
    signInSystem.add_user("john")
    signInSystem.sign_in("john")
    print(signInSystem.all_signed_in())  # False
    signInSystem.sign_in("mike")
    print(signInSystem.all_signed_in())  # False
    signInSystem.sign_in("jack")
    print(signInSystem.all_signed_in())  # True

    # Test all_not_signed_in
    signInSystem.add_user("a")
    signInSystem.add_user("b")
    print(signInSystem.all_not_signed_in())  # ['a', 'b']
    signInSystem.sign_in("a")
    print(signInSystem.all_not_signed_in())  # ['b']