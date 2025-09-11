import socket
import re


class IpUtil:
    """
    This is a class as a tool for IP that can be used to obtain the local IP address, validate its validity, 
    and also provides the functionality to retrieve the corresponding hostname.
    """

    @staticmethod
    def is_valid_ipv4(ip_address):
        """
        Check if the given IP address is a valid IPv4 address.
        :param ip_address: string, the IP address to check
        :return: bool, True if the IP address is valid, False otherwise
        >>> IpUtil.is_valid_ipv4('192.168.0.123')
        True
        >>> IpUtil.is_valid_ipv4('256.0.0.0')
        False
        """
        pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if re.match(pattern, ip_address):
            octets = list(map(int, ip_address.split('.')))
            return all(0 <= octet < 256 for octet in octets)
        return False

    @staticmethod
    def is_valid_ipv6(ip_address):
        """
        Check if the given IP address is a valid IPv6 address.
        :param ip_address: string, the IP address to check
        :return: bool, True if the IP address is valid, False otherwise
        >>> IpUtil.is_valid_ipv6('2001:0db8:85a3:0000:0000:8a2e:0370:7334')
        True
        >>> IpUtil.is_valid_ipv6('2001:0db8:85a3:::8a2e:0370:7334')
        False
        """
        try:
            socket.inet_pton(socket.AF_INET6, ip_address)
            return True
        except socket.error:
            return False

    @staticmethod
    def get_hostname(ip_address):
        """
        Get the hostname associated with the given IP address.
        :param ip_address: string, the IP address to get the hostname for
        :return: string, the hostname associated with the IP address
        >>> IpUtil.get_hostname('110.242.68.3')
        'www.baidu.com'  # This output can vary based on DNS resolution.
        >>> IpUtil.get_hostname('10.0.0.1')
        'localhost'  # This output can also vary.
        """
        try:
            return socket.gethostbyaddr(ip_address)[0]
        except socket.herror:
            return None


if __name__ == "__main__":
    # Test cases for is_valid_ipv4
    print(IpUtil.is_valid_ipv4('192.168.0.123'))  # Expected output: True
    print(IpUtil.is_valid_ipv4('256.0.0.0'))      # Expected output: False
    
    # Test cases for is_valid_ipv6
    print(IpUtil.is_valid_ipv6('2001:0db8:85a3:0000:0000:8a2e:0370:7334'))  # Expected output: True
    print(IpUtil.is_valid_ipv6('2001:0db8:85a3:::8a2e:0370:7334'))          # Expected output: False

    # Test cases for get_hostname
    print(IpUtil.get_hostname('110.242.68.3'))  # Output may vary based on DNS
    print(IpUtil.get_hostname('10.0.0.1'))      # Output may vary based on DNS