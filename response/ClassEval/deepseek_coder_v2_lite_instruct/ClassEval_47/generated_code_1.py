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
        """
        octets = self.ip_address.split('.')
        if len(octets) != 4:
            return False
        for octet in octets:
            if not octet.isdigit() or not 0 <= int(octet) <= 255:
                return False
        return True

    def get_octets(self):
        """
        If the IP address is valid, the list of four decimal numbers separated by "." constituting the IP address is returned; otherwise, an empty list is returned
        :return: list
        """
        if not self.is_valid():
            return []
        return self.ip_address.split('.')

    def get_binary(self):
        """
        If the IP address is valid, return the binary form of the IP address; otherwise, return ''
        :return: string
        """
        if not self.is_valid():
            return ''
        octets = self.ip_address.split('.')
        binary_octets = [bin(int(octet))[2:].zfill(8) for octet in octets]
        return '.'.join(binary_octets)


# Test cases
if __name__ == "__main__":
    # Test the is_valid method
    ipaddress1 = IPAddress("10.10.10.10")
    print(ipaddress1.is_valid())  # True

    ipaddress2 = IPAddress("256.256.256.256")
    print(ipaddress2.is_valid())  # False

    ipaddress3 = IPAddress("10.10.10")
    print(ipaddress3.is_valid())  # False

    ipaddress4 = IPAddress("10.10.10.10.10")
    print(ipaddress4.is_valid())  # False

    # Test the get_octets method
    ipaddress5 = IPAddress("10.10.10.10")
    print(ipaddress5.get_octets())  # ['10', '10', '10', '10']

    ipaddress6 = IPAddress("255.255.255.255")
    print(ipaddress6.get_octets())  # ['255', '255', '255', '255']

    ipaddress7 = IPAddress("192.168.1.1")
    print(ipaddress7.get_octets())  # ['192', '168', '1', '1']

    # Test the get_binary method
    ipaddress8 = IPAddress("10.10.10.10")
    print(ipaddress8.get_binary())  # "00001010.00001010.00001010.00001010"

    ipaddress9 = IPAddress("255.255.255.255")
    print(ipaddress9.get_binary())  # "11111111.11111111.11111111.11111111"

    ipaddress10 = IPAddress("192.168.1.1")
    print(ipaddress10.get_binary())  # "11000000.10101000.00000001.00000001"