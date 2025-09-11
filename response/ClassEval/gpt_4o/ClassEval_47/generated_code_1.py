class IPAddress:
    """
    This is a class to process IP Address, including validating, getting the octets, and obtaining the binary representation of a valid IP address.
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
            if not octet.isdigit():
                return False
            if not (0 <= int(octet) <= 255):
                return False
        return True

    def get_octets(self):
        """
        If the IP address is valid, return the list of four decimal numbers separated by "." constituting the IP address; otherwise, return an empty list
        :return: list
        """
        if self.is_valid():
            return self.ip_address.split('.')
        return []

    def get_binary(self):
        """
        If the IP address is valid, return the binary form of the IP address; otherwise, return ''
        :return: string
        """
        if not self.is_valid():
            return ''
        octets = self.get_octets()
        binary_octets = [format(int(octet), '08b') for octet in octets]
        return '.'.join(binary_octets)

if __name__ == "__main__":
    # Test case for is_valid
    ipaddress = IPAddress("10.10.10.10")
    print(ipaddress.is_valid())  # Expected output: True

    # Test case for get_octets
    print(ipaddress.get_octets())  # Expected output: ["10", "10", "10", "10"]

    # Test case for get_binary
    print(ipaddress.get_binary())  # Expected output: "00001010.00001010.00001010.00001010"

    # Additional test cases
    invalid_ip = IPAddress("256.100.50.25")
    print(invalid_ip.is_valid())  # Expected output: False
    print(invalid_ip.get_octets())  # Expected output: []
    print(invalid_ip.get_binary())  # Expected output: ''