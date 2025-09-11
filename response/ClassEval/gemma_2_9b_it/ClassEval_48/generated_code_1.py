import socket
import netifaces


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
        >>> IpUtil.is_valid_ipv4('192.168.0.123')
        True
        >>> IpUtil.is_valid_ipv4('256.0.0.0')
        False

        """
        try:
            socket.inet_aton(ip_address)
            return True
        except socket.error:
            return False

    @staticmethod
    def is_valid_ipv6(ip_address):
        """
        Check if the given IP address is a valid IPv6 address.
        :param ip_address:string, the IP address to check
        :return:bool, True if the IP address is valid, False otherwise
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
        :param ip_address:string, the IP address to get the hostname for
        :return: string, the hostname associated with the IP address
        >>> IpUtil.get_hostname('110.242.68.3')
        'www.baidu.com'
        >>> IpUtil.get_hostname('10.0.0.1')

        """
        try:
            return socket.gethostbyaddr(ip_address)[0]
        except socket.herror:
            return None



if __name__ == "__main__":
    instance = IpUtil()

    # Test case for is_valid_ipv4
    print(f"Test case for is_valid_ipv4: {instance.is_valid_ipv4('192.168.0.123')}")
    print(f"Test case for is_valid_ipv4: {instance.is_valid_ipv4('256.0.0.0')}")

    # Test case for is_valid_ipv6
    print(f"Test case for is_valid_ipv6: {instance.is_valid_ipv6('2001:0db8:85a3:0000:0000:8a2e:0370:7334')}")
    print(f"Test case for is_valid_ipv6: {instance.is_valid_ipv6('2001:0db8:85a3:::8a2e:0370:7334')}")

    # Test case for get_hostname
    print(f"Test case for get_hostname: {instance.get_hostname('110.242.68.3')}")
    print(f"Test case for get_hostname: {instance.get_hostname('10.0.0.1')}")