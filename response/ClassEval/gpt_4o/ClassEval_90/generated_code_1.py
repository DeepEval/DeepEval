from urllib.parse import urlparse, parse_qs

class URLHandler:
    """
    The class supports to handle URLs, including extracting the scheme, host, path, query parameters, and fragment.
    """

    def __init__(self, url):
        """
        Initialize URLHandler's URL
        """
        self.url = url
        self.parsed_url = urlparse(url)

    def get_scheme(self):
        """
        get the scheme of the URL
        :return: string, If successful, return the scheme of the URL
        >>> urlhandler = URLHandler("https://www.baidu.com/s?wd=aaa&rsv_spt=1#page")
        >>> urlhandler.get_scheme()
        "https"
        """
        return self.parsed_url.scheme

    def get_host(self):
        """
        Get the second part of the URL, which is the host domain name
        :return: string, If successful, return the host domain name of the URL
        >>> urlhandler = URLHandler("https://www.baidu.com/s?wd=aaa&rsv_spt=1#page")
        >>> urlhandler.get_host()
        "www.baidu.com"
        """
        return self.parsed_url.netloc

    def get_path(self):
        """
        Get the third part of the URL, which is the address of the resource
        :return: string, If successful, return the address of the resource of the URL
        >>> urlhandler = URLHandler("https://www.baidu.com/s?wd=aaa&rsv_spt=1#page")
        >>> urlhandler.get_path()
        "/s"
        """
        return self.parsed_url.path

    def get_query_params(self):
        """
        Get the request parameters for the URL
        :return: dict, If successful, return the request parameters of the URL
        >>> urlhandler = URLHandler("https://www.baidu.com/s?wd=aaa&rsv_spt=1#page")
        >>> urlhandler.get_query_params()
        {"wd": "aaa", "rsv_spt": "1"}
        """
        return {key: value[0] for key, value in parse_qs(self.parsed_url.query).items()}

    def get_fragment(self):
        """
        Get the fragment after '#' in the URL
        :return: string, If successful, return the fragment after '#' of the URL
        >>> urlhandler = URLHandler("https://www.baidu.com/s?wd=aaa&rsv_spt=1#page")
        >>> urlhandler.get_fragment()
        "page"
        """
        return self.parsed_url.fragment

if __name__ == "__main__":
    # Test cases
    urlhandler = URLHandler("https://www.baidu.com/s?wd=aaa&rsv_spt=1#page")

    # Testing get_scheme
    output = urlhandler.get_scheme()
    print("Scheme:", output)  # Output: "https"

    # Testing get_host
    output = urlhandler.get_host()
    print("Host:", output)  # Output: "www.baidu.com"

    # Testing get_path
    output = urlhandler.get_path()
    print("Path:", output)  # Output: "/s"

    # Testing get_query_params
    output = urlhandler.get_query_params()
    print("Query Params:", output)  # Output: {"wd": "aaa", "rsv_spt": "1"}

    # Testing get_fragment
    output = urlhandler.get_fragment()
    print("Fragment:", output)  # Output: "page"