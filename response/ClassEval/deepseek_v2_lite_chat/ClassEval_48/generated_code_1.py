import socket
import ipaddress

class IpUtil:
    """
    This is a class as tool for ip that can be used to obtain the local IP address, validate its validity, and also provides the functionality to retrieve the corresponding hostname.
    """

    @staticmethod
    def is_valid_ipv4(ip_address):
        """
        Check if the given IP address is a valid IPv4 address.
        :param ip_address: string, the IP address to check
        :return: bool, True if the IP address is valid, False otherwise
        """
        try:
            ipaddress.IPv4Address(ip_address)
            return True
        except ipaddress.AddressValueError:
            return False

    @staticmethod
    def is_valid_ipv6(ip_address):
        """
        Check if the given IP address is a valid IPv6 address.
        :param ip_address: string, the IP address to check
        :return: bool, True if the IP address is valid, False otherwise
        """
        try:
            ipaddress.IPv6Address(ip_address)
            return True
        except ipaddress.AddressValueError:
            return False

    @staticmethod
    def get_hostname(ip_address):
        """
        Get the hostname associated with the given IP address.
        :param ip_address: string, the IP address to get the hostname for
        :return: string, the hostname associated with the IP address
        """
        try:
            return socket.gethostbyaddr(ip_address)[0]
        except socket.gaierror:
            return "Unknown hostname"

# Test cases
if __name__ == "__main__":
    # Test with a valid IPv4 address
    print(IpUtil.is_valid_ipv4("192.168.1.1"))
    print(IpUtil.is_valid_ipv4("256.0.0.0"))

    print(IpUtil.is_valid_ipv6("2001:0db8:85a3:0000:0000:8a2e:0370:7334"))
    print(IpUtil.is_valid_ipv6("2001:0db8:85a3:::8a2e:0370:7334"))

    print(IpUtil.get_hostname("127.0.0.1"))
    try:
        print(IpUtil.get_hostname("172.16.0.1"))
    except socket.herror:
        print("Unknown hostname")