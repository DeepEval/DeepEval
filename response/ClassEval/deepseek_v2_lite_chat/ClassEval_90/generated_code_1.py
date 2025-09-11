import urllib.parse

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
        """
        parsed_url = urllib.parse.urlparse(self.url)
        return parsed_url.scheme

    def get_host(self):
        """
        Get the second part of the URL, which is the host domain name
        :return: string, If successful, return the host domain name of the URL
        """
        parsed_url = urllib.parse.urlparse(self.url)
        return parsed_url.netloc

    def get_path(self):
        """
        Get the third part of the URL, which is the address of the resource
        :return: string, If successful, return the address of the resource of the URL
        """
        parsed_url = urllib.parse.urlparse(self.url)
        return parsed_url.path

    def get_query_params(self):
        """
        Get the request parameters for the URL
        :return: dict, If successful, return the request parameters of the URL
        """
        parsed_url = urllib.parse.urlparse(self.url)
        query_string = parsed_url.query
        query_params = dict(map(lambda x: x.split("="), query_string.split("&")))
        return query_params

    def get_fragment(self):
        """
        Get the fragment after '#' in the URL
        :return: string, If successful, return the fragment after '#' of the URL
        """
        parsed_url = urllib.parse.urlparse(self.url)
        return parsed_url.fragment


# Test cases
if __name__ == "__main__":
    urlhandler = URLHandler("https://www.baidu.com/s?wd=aaa&rsv_spt=1#page")
    print(urlhandler.get_scheme())  # Should print: https
    print(urlhandler.get_host())    # Should print: www.baidu.com
    print(urlhandler.get_path())    # Should print: /s?wd=aaa&rsv_spt=1#page
    print(urlhandler.get_query_params())  # Should print: {'wd': 'aaa', 'rsv_spt': '1'}
    print(urlhandler.get_fragment())   # Should print: page