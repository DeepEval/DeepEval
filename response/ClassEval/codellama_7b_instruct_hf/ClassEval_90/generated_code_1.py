import urllib.parse

class URLHandler:
    def __init__(self, url):
        self.url = url

    def get_scheme(self):
        return urllib.parse.urlsplit(self.url).scheme

    def get_host(self):
        return urllib.parse.urlsplit(self.url).netloc

    def get_path(self):
        return urllib.parse.urlsplit(self.url).path

    def get_query_params(self):
        return urllib.parse.parse_qs(urllib.parse.urlsplit(self.url).query)

    def get_fragment(self):
        return urllib.parse.urlsplit(self.url).fragment
    
if __name__ == "__main__":
    url = "https://www.example.com/path/to/resource?query=param#fragment"
    handler = URLHandler(url)
    
    print("Scheme:", handler.get_scheme())  # Output: https
    print("Host:", handler.get_host())      # Output: www.example.com
    print("Path:", handler.get_path())      # Output: /path/to/resource
    print("Query Params:", handler.get_query_params())  # Output: {'query': ['param']}
    print("Fragment:", handler.get_fragment())  # Output: fragment