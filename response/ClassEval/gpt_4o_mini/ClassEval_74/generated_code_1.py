class Server:
    """
    This is a class as a server, which handles a white list, message sending and receiving, and information display.
    """

    def __init__(self):
        """
        Initialize the whitelist as an empty list, and initialize the sending and receiving information as empty dictionaries.
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
        self.white_list.append(addr)
        return self.white_list

    def del_white_list(self, addr):
        """
        Remove an address from the whitelist and do nothing if it does not exist
        :param addr: int, address to be deleted
        :return: new whitelist, return False if the address does not exist
        """
        if addr not in self.white_list:
            return False
        self.white_list.remove(addr)
        return self.white_list

    def recv(self, info):
        """
        Receive information containing address and content. If the address is on the whitelist, receive the content; otherwise, do not receive it
        :param info: dict, information dictionary containing address and content
        :return: if successfully received, return the content of the information; otherwise, return False
        """
        addr = info.get("addr")
        if addr in self.white_list:
            self.receive_struct[addr] = info["content"]
            return info["content"]
        return False

    def send(self, info):
        """
        Send information containing address and content
        :param info: dict, information dictionary containing address and content
        :return: if successfully sent, return nothing; otherwise, return a string indicating an error message
        """
        addr = info.get("addr")
        self.send_struct = info
        return None

    def show(self, type):
        """
        Returns struct of the specified type
        :param type: string, the type of struct to be returned, which can be 'send' or 'receive'
        :return: if type is equal to 'send' or 'receive', return the corresponding struct; otherwise, return False
        """
        if type == 'send':
            return self.send_struct
        elif type == 'receive':
            return self.receive_struct
        return False


if __name__ == "__main__":
    server = Server()

    # Test add_white_list
    output = server.add_white_list(88)
    print(output)  # Expected: [88]

    # Test add_white_list with existing address
    output = server.add_white_list(88)
    print(output)  # Expected: False

    # Test del_white_list
    output = server.del_white_list(88)
    print(output)  # Expected: []

    # Test del_white_list with non-existing address
    output = server.del_white_list(88)
    print(output)  # Expected: False

    # Add back the address
    server.add_white_list(88)

    # Test recv with whitelisted address
    output = server.recv({"addr": 88, "content": "abc"})
    print(output)  # Expected: "abc"

    # Test recv with non-whitelisted address
    output = server.recv({"addr": 66, "content": "xyz"})
    print(output)  # Expected: False

    # Test send
    output = server.send({"addr": 66, "content": "ABC"})
    print(output)  # Expected: None

    # Test show for send
    output = server.show("send")
    print(output)  # Expected: {"addr": 66, "content": "ABC"}

    # Test show for receive
    output = server.show("receive")
    print(output)  # Expected: {88: "abc"}