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
        return self.parsed_url.path + ('?' + self.parsed_url.query if self.parsed_url.query else '') + ('#' + self.parsed_url.fragment if self.parsed_url.fragment else '')

    def get_query_params(self):
        """
        Get the request parameters for the URL
        :return: dict, If successful, return the request parameters of the URL
        >>> urlhandler = URLHandler("https://www.baidu.com/s?wd=aaa&rsv_spt=1#page")
        >>> urlhandler.get_query_params()
        {"wd": ["aaa"], "rsv_spt": ["1"]}
        """
        return parse_qs(self.parsed_url.query)

    def get_fragment(self):
        """
        Get the fragment after '#' in the URL
        :return: string, If successful, return the fragment after '#' of the URL
        >>> urlhandler = URLHandler("https://www.baidu.com/s?wd=aaa&rsv_spt=1#page")
        >>> urlhandler.get_fragment()
        "page"
        """
        return self.parsed_url.fragment


# Test cases
if __name__ == "__main__":
    urlhandler = URLHandler("https://www.baidu.com/s?wd=aaa&rsv_spt=1#page")
    
    # Test get_scheme
    output_scheme = urlhandler.get_scheme()
    print(f"Scheme: {output_scheme}")  # Expected output: "https"
    
    # Test get_host
    output_host = urlhandler.get_host()
    print(f"Host: {output_host}")  # Expected output: "www.baidu.com"
    
    # Test get_path
    output_path = urlhandler.get_path()
    print(f"Path: {output_path}")  # Expected output: "/s?wd=aaa&rsv_spt=1#page"
    
    # Test get_query_params
    output_query_params = urlhandler.get_query_params()
    print(f"Query Params: {output_query_params}")  # Expected output: {"wd": ["aaa"], "rsv_spt": ["1"]}
    
    # Test get_fragment
    output_fragment = urlhandler.get_fragment()
    print(f"Fragment: {output_fragment}")  # Expected output: "page"