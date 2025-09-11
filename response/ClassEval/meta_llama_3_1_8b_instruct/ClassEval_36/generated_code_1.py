import datetime

class EmailClient:
    """
    This is a class that serves as an email client, implementing functions such
    as checking emails, determining whether there is sufficient space, and cleaning
    up space
    """

    def __init__(self, addr, capacity) -> None:
        """
        Initializes the EmailClient class with the email address and the capacity
        of the email box.
        :param addr: The email address, str.
        :param capacity: The capacity of the email box, float.
        """
        self.addr = addr
        self.capacity = capacity
        self.inbox = []

    def send_to(self, recv, content, size):
        """
        Sends an email to the given email address.
        :param recv: The email address of the receiver, str.
        :param content: The content of the email, str.
        :param size: The size of the email, float.
        :return: True if the email is sent successfully, False if the receiver's
        email box is full.
        """
        if size <= self.capacity - self.get_occupied_size():
            email = {
               'sender': self.addr,
               'receiver': recv,
                'content': content,
               'size': size,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               'state': 'unread'
            }
            self.inbox.append(email)
            return True
        else:
            return False

    def fetch(self):
        """
        Retrieves the first unread email in the email box and marks it as read.
        :return: The first unread email in the email box, dict.
        """
        if self.inbox:
            email = self.inbox[0]
            email['state'] ='read'
            return email
        else:
            return None

    def is_full_with_one_more_email(self, size):
        """
        Determines whether the email box is full after adding an email of the given
        size.
        :param size: The size of the email, float.
        :return: True if the email box is full, False otherwise.
        """
        return self.get_occupied_size() + size > self.capacity

    def get_occupied_size(self):
        """
        Gets the total size of the emails in the email box.
        :return: The total size of the emails in the email box, float.
        """
        return sum(email['size'] for email in self.inbox)

    def clear_inbox(self, size):
        """
        Clears the email box by deleting the oldest emails until the email box
        has enough space to accommodate the given size.
        :param size: The size of the email, float.
        """
        while self.get_occupied_size() > self.capacity - size:
            self.inbox.pop(0)

if __name__ == "__main__":
    sender = EmailClient('sender@example.com', 100)
    receiver = EmailClient('receiver@example.com', 50)

    # Test case for send_to method
    print(receiver.send_to('sender@example.com', 'Hello', 10))  # Expected output: True
    print(receiver.inbox)  # Expected output: [{'sender':'sender@example.com','receiver':'receiver@example.com', 'content': 'Hello','size': 10, 'time': '2023-07-13 11:36:40','state': 'unread'}]
    print(receiver.send_to('sender@example.com', 'Hello2', 40))  # Expected output: False

    # Test case for fetch method
    print(receiver.fetch())  # Expected output: {'sender':'sender@example.com','receiver':'receiver@example.com', 'content': 'Hello','size': 10, 'time': '2023-07-13 11:36:40','state':'read'}

    # Test case for is_full_with_one_more_email method
    print(receiver.is_full_with_one_more_email(40))  # Expected output: True

    # Test case for get_occupied_size method
    print(receiver.get_occupied_size())  # Expected output: 10

    # Test case for clear_inbox method
    receiver.clear_inbox(30)
    print(receiver.inbox)  # Expected output: [{'sender':'sender@example.com','receiver':'receiver@example.com', 'content': 'Hello','size': 10, 'time': '2023-07-13 11:36:40','state':'read'}]