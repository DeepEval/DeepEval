from datetime import datetime

class EmailClient:
    """
    This is a class that serves as an email client, implementing functions such as checking emails, 
    determining whether there is sufficient space, and cleaning up space
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
        # Assuming the email is sent successfully if there is enough space in the inbox
        if sum([email['size'] for email in self.inbox]) + size <= self.capacity:
            self.inbox.append({'sender': recv, 'receiver': self.addr, 'content': content, 'size': size, 'time': datetime.now(), 'state': 'unread'})
            return True
        else:
            return False

    def fetch(self):
        """
        Retrieves the first unread email in the email box and marks it as read.
        :return: The first unread email in the email box, dict.
        """
        # Assuming the oldest unread email is read
        self.inbox[0]['state'] = 'read'
        return self.inbox.pop(0)

    def is_full_with_one_more_email(self, size):
        """
        Determines whether the email box is full after adding an email of the given size.
        :param size: The size of the email, float.
        :return: True if the email box is full, False otherwise.
        """
        # Checks if adding an email of given size would exceed the capacity
        if sum([email['size'] for email in self.inbox]) + size > self.capacity:
            return True
        else:
            return False

    def get_occupied_size(self):
        """
        Gets the total size of the emails in the email box.
        :return: The total size of the emails in the email box, float.
        """
        # Returns the total size of all emails in the inbox
        return sum([email['size'] for email in self.inbox])

    def clear_inbox(self, size):
        """
        Clears the email box by deleting the oldest emails until the email box has enough space to accommodate the given size.
        :param size: The size of the email, float.
        """
        # Deletes emails from the inbox until there is enough space, starting with the oldest
        while sum([email['size'] for email in self.inbox]) > self.capacity - size:
            self.inbox.pop(0)


if __name__ == "__main__":
    
    alice = EmailClient("alice@example.com", 100.0)
    bob = EmailClient("bob@example.com", 200.0)

    
    print("=== 测试发送邮件 ===")
    send_result = alice.send_to("bob@example.com", "Hello Bob!", 30.0)
    print(f"Alice发送邮件结果: {'成功' if send_result else '失败'}") 
    print(f"Alice邮箱已用空间: {alice.get_occupied_size()}") 

    send_result = bob.send_to("alice@example.com", "Hi Alice!", 80.0)
    print(f"Bob发送邮件结果: {'成功' if send_result else '失败'}")  
    print(f"Bob邮箱已用空间: {bob.get_occupied_size()}")  

    send_result = alice.send_to("bob@example.com", "Another email", 80.0)
    print(f"Alice再次发送邮件结果: {'成功' if send_result else '失败'}")  
    print(f"Alice邮箱已用空间: {alice.get_occupied_size()}")  


    print("\n=== 测试收取邮件 ===")

    received = alice.fetch()
    print(f"Alice收取的邮件: {received['content']}，发送者: {received['sender']}")
    print(f"Alice邮箱剩余邮件数: {len(alice.inbox)}")  


    print("\n=== 测试邮箱是否已满 ===")

    is_full = bob.is_full_with_one_more_email(130.0)
    print(f"Bob邮箱添加130大小邮件后是否已满: {'是' if is_full else '否'}")


    print("\n=== 测试清理空间 ===")

    bob.send_to("alice@example.com", "Third email", 50.0)
    print(f"Bob清理前已用空间: {bob.get_occupied_size()}") 


    bob.clear_inbox(100.0)
    print(f"Bob清理后已用空间: {bob.get_occupied_size()}")  
    print(f"Bob清理后邮件数: {len(bob.inbox)}") 