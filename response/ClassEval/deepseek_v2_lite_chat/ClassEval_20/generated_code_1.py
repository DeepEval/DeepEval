from datetime import datetime

class Chat:
    """
    This is a chat class with the functions of adding users, removing users, sending messages, and obtaining messages.
    """

    def __init__(self):
        """
        Initialize the Chat with an attribute users, which is an empty dictionary.
        """
        self.users = {}

    def add_user(self, username):
        """
        Add a new user to the Chat.
        :param username: The user's name, str.
        :return: If the user is already in the Chat, returns False, otherwise, returns True.
        """
        if username in self.users:
            return False
        self.users[username] = []
        return True

    def remove_user(self, username):
        """
        Remove a user from the Chat.
        :param username: The user's name, str.
        :return: If the user is already in the Chat, returns True, otherwise, returns False.
        """
        if username not in self.users:
            return False
        del self.users[username]
        return True

    def send_message(self, sender, receiver, message):
        """
        Send a message from a user to another user.
        :param sender: The sender's name, str.
        :param receiver: The receiver's name, str.
        :param message: The message, str.
        :return: If the sender or the receiver is not in the Chat, returns False, otherwise, returns True.
        """
        if sender not in self.users or receiver not in self.users:
            return False
        message_dict = {'sender': sender, 'receiver': receiver, 'message': message, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        self.users[sender].append(message_dict)
        return True

    def get_messages(self, username):
        """
        Get all the messages of a user from the Chat.
        :param username: The user's name, str.
        :return: A list of messages, each message is a dictionary with keys 'sender', 'receiver', 'message', 'timestamp'.
        """
        if username not in self.users:
            return []
        return self.users[username]

if __name__ == "__main__":
    chat = Chat()
    
    # Test case for add_user
    print(chat.add_user('John'))  # True
    print(chat.add_user('John'))  # False

    # Test case for remove_user
    chat.users = {'John': [{'sender': 'John', 'receiver': 'Mary', 'message': 'Hello', 'timestamp': '2023-01-01 00:00:00'}]}
    print(chat.remove_user('John'))  # True
    print(chat.remove_user('John'))  # False

    # Test case for send_message
    chat.users = {'John': [], 'Mary': []}
    print(chat.send_message('John', 'Mary', 'Hello'))  # True
    print(chat.send_message('John', 'Tom', 'Hello'))  # False

    # Test case for get_messages
    chat.users = {'John': [{'sender': 'John', 'receiver': 'Mary', 'message': 'Hello', 'timestamp': '2023-01-01 00:00:00'}]}
    print(chat.get_messages('John'))  # [{'sender': 'John', 'receiver': 'Mary', 'message': 'Hello', 'timestamp': '2023-01-01 00:00:00'}]
    print(chat.get_messages('Mary'))  # []