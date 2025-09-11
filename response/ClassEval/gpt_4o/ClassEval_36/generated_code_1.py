from datetime import datetime

class EmailClient:
    """
    This is a class that serves as an email client, implementing functions such as checking emails, 
    determining whether there is sufficient space, and cleaning up space.
    """

    def __init__(self, addr, capacity) -> None:
        """
        Initializes the EmailClient class with the email address and the capacity of the email box.
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
        :return: True if the email is sent successfully, False if the receiver's email box is full.
        """
        if recv.is_full_with_one_more_email(size):
            return False
        
        email = {
            'sender': self.addr,
            'receiver': recv.addr,
            'content': content,
            'size': size,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'state': 'unread'
        }
        recv.inbox.append(email)
        return True

    def fetch(self):
        """
        Retrieves the first unread email in the email box and marks it as read.
        :return: The first unread email in the email box, dict.
        """
        for email in self.inbox:
            if email['state'] == 'unread':
                email['state'] = 'read'
                return email
        return None

    def is_full_with_one_more_email(self, size):
        """
        Determines whether the email box is full after adding an email of the given size.
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
        Clears the email box by deleting the oldest emails until the email box has enough space to accommodate the given size.
        :param size: The size of the email, float.
        """
        while self.is_full_with_one_more_email(size):
            if self.inbox:
                self.inbox.pop(0)
            else:
                break


if __name__ == "__main__":
    # Test case for send_to method
    sender = EmailClient('sender@example.com', 100)
    receiver = EmailClient('receiver@example.com', 50)
    output = sender.send_to(receiver, 'Hello', 10)
    print(output)  # Expected: True
    print(receiver.inbox)

    # Test case for fetch method
    fetched_email = receiver.fetch()
    print(fetched_email)

    # Test case for is_full_with_one_more_email method
    print(receiver.is_full_with_one_more_email(10))  # Expected: False

    # Test case for get_occupied_size method
    sender.inbox = [{'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 10, 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'state': 'unread'}]
    print(sender.get_occupied_size())  # Expected: 10

    # Test case for clear_inbox method
    receiver.inbox = [{'size': 10},{'size': 20},{'size': 15}]
    receiver.clear_inbox(30)
    print(receiver.inbox)  # Expected: [{'size': 15}]