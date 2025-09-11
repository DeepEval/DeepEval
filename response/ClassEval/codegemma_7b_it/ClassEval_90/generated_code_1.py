import re

class URLHandler:
    """
    The class supports to handle URLs, including extracting the scheme, host, path, query parameters, and fragment.
    """

    def __init__(self, url):
        """
        Initialize URLHandler's URL
        """
        self.url = url

    def get_scheme(self):
        """
        get the scheme of the URL
        :return: string, If successful, return the scheme of the URL
        >>> urlhandler = URLHandler("https://www.baidu.com/s?wd=aaa&rsv_spt=1#page")
        >>> urlhandler.get_scheme()
        "https"
        """
        match = re.search(r'(^[a-zA-Z]+):', self.url)
        if match:
            return match.group(1)
        else:
            return None

    def get_host(self):
        """
        Get the second part of the URL, which is the host domain name
        :return: string, If successful, return the host domain name of the URL
        >>> urlhandler = URLHandler("https://www.baidu.com/s?wd=aaa&rsv_spt=1#page")
        >>> urlhandler.get_host()
        "www.baidu.com"
        """
        match = re.search(r'(?:[a-zA-Z]+:)?//([^/]+)', self.url)
        if match:
            return match.group(1)
        else:
            return None

    def get_path(self):
        """
        Get the third part of the URL, which is the address of the resource
        :return: string, If successful, return the address of the resource of the URL
        >>> urlhandler = URLHandler("https://www.baidu.com/s?wd=aaa&rsv_spt=1#page")
        >>> urlhandler.get_path()
        "/s?wd=aaa&rsv_spt=1#page"
        """
        match = re.search(r'(?:[a-zA-Z]+:)?//[^/]+([^#]+)', self.url)
        if match:
            return match.group(1)
        else:
            return None

    def get_query_params(self):
        """
        Get the request parameters for the URL
        :return: dict, If successful, return the request parameters of the URL
        >>> urlhandler = URLHandler("https://www.baidu.com/s?wd=aaa&rsv_spt=1#page")
        >>> urlhandler.get_query_params()
        {"wd": "aaa", "rsv_spt": "1"}
        """
        query_params = {}
        match = re.search(r'\?(.*)', self.url)
        if match:
            query_string = match.group(1)
            for param in query_string.split('&'):
                key, value = param.split('=')
                query_params[key] = value
        return query_params

    def get_fragment(self):
        """
        Get the fragment after '#' in the URL
        :return: string, If successful, return the fragment after '#' of the URL
        >>> urlhandler = URLHandler("https://www.baidu.com/s?wd=aaa&rsv_spt=1#page")
        >>> urlhandler.get_fragment()
        "page"
        """
        match = re.search(r'#[^#]+', self.url)
        if match:
            return match.group(0)[1:]
        else:
            return None

if __name__ == "__main__":
    urlhandler = URLHandler("https://www.baidu.com/s?wd=aaa&rsv_spt=1#page")

    print(urlhandler.get_scheme())  # Output: https
    print(urlhandler.get_host())  # Output: www.baidu.com
    print(urlhandler.get_path())  # Output: /s?wd=aaa&rsv_spt=1#page
    print(urlhandler.get_query_params())  # Output: {'wd': 'aaa', 'rsv_spt': '1'}
    print(urlhandler.get_fragment())  # Output: page