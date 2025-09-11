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
        Get the scheme of the URL
        :return: string, If successful, return the scheme of the URL
        """
        return urllib.parse.urlparse(self.url).scheme

    def get_host(self):
        """
        Get the second part of the URL, which is the host domain name
        :return: string, If successful, return the host domain name of the URL
        """
        return urllib.parse.urlparse(self.url).netloc

    def get_path(self):
        """
        Get the third part of the URL, which is the address of the resource
        :return: string, If successful, return the address of the resource of the URL
        """
        return urllib.parse.urlparse(self.url).path

    def get_query_params(self):
        """
        Get the request parameters for the URL
        :return: dict, If successful, return the request parameters of the URL
        """
        query_params = urllib.parse.parse_qs(urllib.parse.urlparse(self.url).query)
        return {key: value[0] for key, value in query_params.items()}

    def get_fragment(self):
        """
        Get the fragment after '#' in the URL
        :return: string, If successful, return the fragment after '#' of the URL
        """
        return urllib.parse.urlparse(self.url).fragment

if __name__ == "__main__":
    urlhandler = URLHandler("https://www.baidu.com/s?wd=aaa&rsv_spt=1#page")
    
    print("Scheme:", urlhandler.get_scheme())
    print("Host:", urlhandler.get_host())
    print("Path:", urlhandler.get_path())
    print("Query Parameters:", urlhandler.get_query_params())
    print("Fragment:", urlhandler.get_fragment())