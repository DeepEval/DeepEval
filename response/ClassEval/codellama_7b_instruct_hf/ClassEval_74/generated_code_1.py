import collections

class Server:
    def __init__(self):
        self.white_list = []
        self.send_struct = {}
        self.receive_struct = {}

    def add_white_list(self, addr):
        if addr not in self.white_list:
            self.white_list.append(addr)
        return self.white_list

    def del_white_list(self, addr):
        if addr in self.white_list:
            self.white_list.remove(addr)
        return self.white_list

    def recv(self, info):
        if info["addr"] in self.white_list:
            return info["content"]
        else:
            return False

    def send(self, info):
        self.send_struct = info
        return None

    def show(self, type):
        if type == "send":
            return self.send_struct
        elif type == "receive":
            return self.receive_struct
        else:
            return False

if __name__ == "__main__":
    server = Server()

    # Test add_white_list method
    print(server.add_white_list(88))
    # Output: [88]

    # Test del_white_list method
    print(server.del_white_list(88))
    # Output: []

    # Test recv method
    info = {"addr": 88, "content": "abc"}
    print(server.recv(info))
    # Output: abc

    # Test send method
    info = {"addr": 66, "content": "ABC"}
    print(server.send(info))
    # Output: None

    # Test show method
    print(server.show("send"))
    # Output: {'addr': 66, 'content': 'ABC'}

    print(server.show("receive"))
    # Output: {}