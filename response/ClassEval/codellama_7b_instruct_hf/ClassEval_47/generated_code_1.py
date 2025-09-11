import ipaddress

class IPAddress:
    def __init__(self, ip_address):
        self.ip_address = ip_address

    def is_valid(self):
        try:
            ipaddress.ip_address(self.ip_address)
            return True
        except ValueError:
            return False

    def get_octets(self):
        if self.is_valid():
            ip_address = ipaddress.ip_address(self.ip_address)
            return [str(ip_address.ipv4_address[i]) for i in range(4)]
        else:
            return []

    def get_binary(self):
        if self.is_valid():
            ip_address = ipaddress.ip_address(self.ip_address)
            return ip_address.packed.hex()
        else:
            return ''

if __name__ == "__main__":
    # Test cases
    ipaddress = IPAddress("10.10.10.10")
    print(ipaddress.is_valid())  # True
    print(ipaddress.get_octets())  # ["10", "10", "10", "10"]
    print(ipaddress.get_binary())  # "00001010.00001010.00001010.00001010"

    ipaddress = IPAddress("127.0.0.1")
    print(ipaddress.is_valid())  # True
    print(ipaddress.get_octets())  # ["127", "0", "0", "1"]
    print(ipaddress.get_binary())  # "01111111.00000000.00000000.00000001"

    ipaddress = IPAddress("10.10.10.256")
    print(ipaddress.is_valid())  # False
    print(ipaddress.get_octets())  # []
    print(ipaddress.get_binary())  # ""