import ipaddress

class IPAddress:
    """
    This is a class to process IP Address, including validating, getting the octets and obtaining the binary representation of a valid IP address.
    """

    def __init__(self, ip_address):
        """
        Initialize the IP address to the specified address
        :param ip_address: string
        """
        self.ip_address = ip_address


    def is_valid(self):
        """
        Judge whether the IP address is valid, that is, whether the IP address is composed of four Decimal digits separated by '.'. Each digit is greater than or equal to 0 and less than or equal to 255
        :return: bool
        >>> ipaddress = IPAddress("10.10.10.10")
        >>> ipaddress.is_valid()
        True
        """
        try:
            ipaddress.ip_address(self.ip_address)
            return True
        except ValueError:
            return False


    def get_octets(self):
        """
        If the IP address is valid, the list of four decimal numbers separated by "." constituting the IP address is returned; otherwise, an empty list is returned
        :return: list
        >>> ipaddress = IPAddress("10.10.10.10")
        >>> ipaddress.get_octets()
        ["10", "10", "10", "10"]
        """
        if self.is_valid():
            return [str(octet) for octet in self.ip_address.split('.')]
        else:
            return []


    def get_binary(self):
        """
        If the IP address is valid, return the binary form of the IP address; otherwise, return ''
        :return: string
        >>> ipaddress = IPAddress("10.10.10.10")
        >>> ipaddress.get_binary()
        "00001010.00001010.00001010.00001010"
        """
        if self.is_valid():
            return '.'.join([bin(int(octet))[2:].zfill(8) for octet in self.ip_address.split('.')])
        else:
            return ''

if __name__ == "__main__":
    # Test cases
    instance = IPAddress("10.10.10.10")
    print("is_valid():", instance.is_valid())
    print("get_octets():", instance.get_octets())
    print("get_binary():", instance.get_binary())

    instance = IPAddress("255.255.255.255")
    print("is_valid():", instance.is_valid())
    print("get_octets():", instance.get_octets())
    print("get_binary():", instance.get_binary())

    instance = IPAddress("10.10.10")
    print("is_valid():", instance.is_valid())
    print("get_octets():", instance.get_octets())
    print("get_binary():", instance.get_binary())

    instance = IPAddress("10.10.10.256")
    print("is_valid():", instance.is_valid())
    print("get_octets():", instance.get_octets())
    print("get_binary():", instance.get_binary())

    instance = IPAddress("abc.def.ghi.jkl")
    print("is_valid():", instance.is_valid())
    print("get_octets():", instance.get_octets())
    print("get_binary():", instance.get_binary())