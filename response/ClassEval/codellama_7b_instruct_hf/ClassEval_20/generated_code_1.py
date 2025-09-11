import datetime

class Chat:
    def __init__(self):
        self.users = {}

    def add_user(self, username):
        if username in self.users:
            return False
        else:
            self.users[username] = []
            return True

    def remove_user(self, username):
        if username not in self.users:
            return False
        else:
            del self.users[username]
            return True

    def send_message(self, sender, receiver, message):
        if sender not in self.users or receiver not in self.users:
            return False
        else:
            self.users[sender].append({
                'sender': sender,
                'receiver': receiver,
                'message': message,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            return True

    def get_messages(self, username):
        if username not in self.users:
            return []
        else:
            return self.users[username]

if __name__ == "__main__":
    chat = Chat()

    # Test add_user
    print("Test add_user:")
    print(chat.add_user("John"))  # Output: True
    print(chat.add_user("John"))  # Output: False

    # Test remove_user
    print("Test remove_user:")
    print(chat.remove_user("John"))  # Output: True
    print(chat.remove_user("John"))  # Output: False

    # Test send_message
    print("Test send_message:")
    print(chat.send_message("John", "Mary", "Hello"))  # Output: True
    print(chat.send_message("John", "Tom", "Hello"))  # Output: False

    # Test get_messages
    print("Test get_messages:")
    print(chat.get_messages("John"))  # Output: [{'sender': 'John', 'receiver': 'Mary', 'message': 'Hello', 'timestamp': '2023-01-01 00:00:00'}]
    print(chat.get_messages("Mary"))  # Output: []