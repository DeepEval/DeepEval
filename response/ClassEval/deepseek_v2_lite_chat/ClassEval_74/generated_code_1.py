class Server:
    def __init__(self):
        self.white_list = []
        self.send_struct = {}
        self.receive_struct = {}

    def add_white_list(self, addr):
        if addr not in self.white_list:
            self.white_list.append(addr)
            return self.white_list
        return False

    def del_white_list(self, addr):
        if addr in self.white_list:
            self.white_list.remove(addr)
            return self.white_list
        return False

    def recv(self, info):
        addr = info['addr']
        content = info['content']
        if addr in self.white_list:
            self.receive_struct[addr] = content
            return content
        return False

    def send(self, info):
        addr = info['addr']
        content = info['content']
        if addr not in self.send_struct:
            self.send_struct[addr] = content
            return
        return "Error: Address already in the send structure."

    def show(self, type):
        if type == 'send' or type == 'receive':
            return self.send_struct if type == 'send' else self.receive_struct
        return False

# Test cases
if __name__ == "__main__":
    server = Server()

    # Test add_white_list
    print(server.add_white_list(88))  # Expected: [88]
    print(server.add_white_list(88))  # Expected: False (already exists)

    # Test del_white_list
    print(server.del_white_list(88))  # Expected: []
    print(server.del_white_list(88))  # Expected: False (does not exist)

    # Test recv
    print(server.recv({"addr": 88, "content": "abc"}))  # Expected: abc
    print(server.recv({"addr": 88, "content": "def"}))  # Expected: False (not on whitelist)

    # Test send
    print(server.send({"addr": 66, "content": "ABC"}))  # Expected: {"addr": 66, "content": "ABC"}
    print(server.send({"addr": 66, "content": "DEF"}))  # Expected: 'Error: Address already in the send structure.'

    # Test show
    print(server.show('send'))  # Expected: {"addr": 66, "content": "ABC"}
    print(server.show('receive'))  # Expected: {"addr": 88, "content": "abc"}
    print(server.show('unknown'))  # Expected: False (invalid type)