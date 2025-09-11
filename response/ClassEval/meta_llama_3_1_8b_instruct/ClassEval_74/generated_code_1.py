import unittest

class Server:
    """
    This is a class as a server, which handles a white list, message sending and
    receiving, and information display.
    """

    def __init__(self):
        """
        Initialize the whitelist as an empty list, and initialize the sending and
        receiving information as an empty dictionary
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
        if addr not in self.white_list:
            self.white_list.append(addr)
            return self.white_list
        else:
            return False


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
        Receive information containing address and content. If the address is on
        the whitelist, receive the content; otherwise, do not receive it
        :param info: dict, information dictionary containing address and content
        :return: if successfully received, return the content of the infomation;
        otherwise, return False
        """
        if info['addr'] in self.white_list:
            self.receive_struct = info
            return info['content']
        else:
            return False


    def send(self, info):
        """
        Send information containing address and content
        :param info: dict, information dictionary containing address and content
        :return: if successfully sent, return nothing; otherwise, return a string
        indicating an error message
        """
        if isinstance(info, dict) and 'addr' in info and 'content' in info:
            self.send_struct = info
            return
        else:
            return "Error: Invalid input information"


    def show(self, type):
        """
        Returns struct of the specified type
        :param type: string, the type of struct to be returned, which can be 'send'
        or'receive'
        :return: if type is equal to 'send' or'receive', return the corresponding
        struct; otherwise, return False
        """
        if type =='send':
            return self.send_struct
        elif type =='receive':
            return self.receive_struct
        else:
            return False


class TestServer(unittest.TestCase):

    def test_add_white_list(self):
        server = Server()
        output = server.add_white_list(88)
        self.assertEqual(output, [88])

    def test_del_white_list(self):
        server = Server()
        server.add_white_list(88)
        output = server.del_white_list(88)
        self.assertEqual(output, [])

    def test_recv(self):
        server = Server()
        server.add_white_list(88)
        output = server.recv({"addr":88,"content":"abc"})
        self.assertEqual(output, "abc")

    def test_send(self):
        server = Server()
        server.send({"addr":66,"content":"ABC"})
        self.assertEqual(server.send_struct, {"addr":66,"content":"ABC"})

    def test_show(self):
        server = Server()
        server.send({"addr":66,"content":"ABC"})
        server.recv({"addr":88,"content":"abc"})
        output = server.show("send")
        self.assertEqual(output, {"addr":66,"content":"ABC"})
        output = server.show("receive")
        self.assertEqual(output, {"addr":88,"content":"abc"})

if __name__ == "__main__":
    server = Server()
    server.add_white_list(88)
    server.send({"addr":66,"content":"ABC"})
    server.recv({"addr":88,"content":"abc"})
    print(server.show("send"))
    print(server.show("receive"))