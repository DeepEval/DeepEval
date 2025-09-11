class Server:
    """
    This is a class as a server, which handles a whitelist, message sending and receiving, and information display.
    """

    def __init__(self):
        """
        Initialize the whitelist as an empty list, and initialize the sending and receiving information as an empty dictionary
        """
        self.white_list = []
        self.send_struct = {}
        self.receive_struct = {}

    def add_white_list(self, addr):
        """
        Add an address to the whitelist and do nothing if it already exists
        :param addr: int, address to be added
        :return: new whitelist, return False if the address already exists
        """
        if addr in self.white_list:
            return False
        else:
            self.white_list.append(addr)
            return self.white_list

    def del_white_list(self, addr):
        """
        Remove an address from the whitelist and do nothing if it does not exist
        :param addr: int, address to be deleted
        :return: new whitelist, return False if the address does not exist
        """
        if addr in self.white_list:
            self.white_list.remove(addr)
            return self.white_list
        else:
            return False

    def recv(self, info):
        """
        Receive information containing address and content. If the address is on the whitelist, receive the content; otherwise, do not receive it
        :param info: dict, information dictionary containing address and content
        :return: if successfully received, return the content of the information; otherwise, return False
        """
        addr = info.get("addr")
        content = info.get("content")
        if addr in self.white_list:
            self.receive_struct = info
            return content
        else:
            return False

    def send(self, info):
        """
        Send information containing address and content
        :param info: dict, information dictionary containing address and content
        :return: if successfully sent, return nothing; otherwise, return a string indicating an error message
        """
        addr = info.get("addr")
        content = info.get("content")
        self.send_struct = {"addr": addr, "content": content}

    def show(self, type):
        """
        Returns struct of the specified type
        :param type: string, the type of struct to be returned, which can be 'send' or 'receive'
        :return: if type is equal to 'send' or 'receive', return the corresponding struct; otherwise, return False
        """
        if type == "send":
            return self.send_struct
        elif type == "receive":
            return self.receive_struct
        else:
            return False


if __name__ == "__main__":
    server = Server()

    # Test add_white_list method
    output = server.add_white_list(88)
    print(output)  # Expected: [88]

    # Test del_white_list method
    server.add_white_list(88)
    output = server.del_white_list(88)
    print(output)  # Expected: []

    # Test recv method
    server.add_white_list(88)
    output = server.recv({"addr": 88, "content": "abc"})
    print(output)  # Expected: 'abc'

    # Test send method
    server.send({"addr": 66, "content": "ABC"})
    print(server.send_struct)  # Expected: {'addr': 66, 'content': 'ABC'}

    # Test show method
    server.recv({"addr": 88, "content": "abc"})
    server.send({"addr": 66, "content": "ABC"})
    output = server.show("send")
    print(output)  # Expected: {'addr': 66, 'content': 'ABC'}