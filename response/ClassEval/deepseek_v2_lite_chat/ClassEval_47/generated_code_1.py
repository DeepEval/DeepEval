import re

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
        pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        return bool(re.match(pattern, self.ip_address))

    def get_octets(self):
        """
        If the IP address is valid, the list of four decimal numbers separated by "." constituting the IP address is returned; otherwise, an empty list is returned
        :return: list
        """
        if not self.is_valid():
            return []
        octets = self.ip_address.split('.')
        return [octet for octet in octets if octet]  # Remove empty entries if any

    def get_binary(self):
        """
        If the IP address is valid, return the binary form of the IP address; otherwise, return ''
        :return: string
        """
        if not self.is_valid():
            return ''
        octets = self.get_octets()
        binary_parts = [bin(int(octet))[2:].zfill(8) for octet in octets]
        return '.'.join(binary_parts)

# Test cases
if __name__ == "__main__":
    ipaddress = IPAddress("10.10.10.10")
    print(ipaddress.is_valid())  # True
    print(ipaddress.get_octets())  # ['10', '10', '10', '10']
    print(ipaddress.get_binary())  # '00001010.00001010.00001010.00001010'

    ipaddress = IPAddress("256.10.10.10")  # Invalid IP
    print(ipaddress.is_valid())  # False
    print(ipaddress.get_octets())  # []
    print(ipaddress.get_binary())  # ''