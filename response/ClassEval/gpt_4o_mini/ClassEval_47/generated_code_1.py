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
        Judge whether the IP address is valid, that is, whether the IP address is composed of four Decimal digits separated by '.'. 
        Each digit is greater than or equal to 0 and less than or equal to 255
        :return: bool
        >>> ipaddress = IPAddress("10.10.10.10")
        >>> ipaddress.is_valid()
        True
        """
        pattern = r'^([0-2]?[0-5]{0,2}|[1]?[0-9]{1,2}|[1-9][0-9]?)\.([0-2]?[0-5]{0,2}|[1]?[0-9]{1,2}|[1-9][0-9]?)\.([0-2]?[0-5]{0,2}|[1]?[0-9]{1,2}|[1-9][0-9]?)\.([0-2]?[0-5]{0,2}|[1]?[0-9]{1,2}|[1-9][0-9]?)$'
        return bool(re.match(pattern, self.ip_address))

    def get_octets(self):
        """
        If the IP address is valid, the list of four decimal numbers separated by "." constituting the IP address is returned; 
        otherwise, an empty list is returned
        :return: list
        >>> ipaddress = IPAddress("10.10.10.10")
        >>> ipaddress.get_octets()
        ['10', '10', '10', '10']
        """
        if self.is_valid():
            return self.ip_address.split('.')
        return []

    def get_binary(self):
        """
        If the IP address is valid, return the binary form of the IP address; otherwise, return ''
        :return: string
        >>> ipaddress = IPAddress("10.10.10.10")
        >>> ipaddress.get_binary()
        '00001010.00001010.00001010.00001010'
        """
        if self.is_valid():
            return '.'.join(f'{int(octet):08b}' for octet in self.get_octets())
        return ''

# Test cases
if __name__ == "__main__":
    ipaddress = IPAddress("10.10.10.10")
    
    # Test is_valid method
    valid_output = ipaddress.is_valid()
    print(f"is_valid: {valid_output}")  # Expected output: True

    # Test get_octets method
    octets_output = ipaddress.get_octets()
    print(f"get_octets: {octets_output}")  # Expected output: ['10', '10', '10', '10']

    # Test get_binary method
    binary_output = ipaddress.get_binary()
    print(f"get_binary: {binary_output}")  # Expected output: '00001010.00001010.00001010.00001010'