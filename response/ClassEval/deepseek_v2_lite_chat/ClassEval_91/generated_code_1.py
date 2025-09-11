import urllib.parse

class UrlPath:
    """
    The class is a utility for encapsulating and manipulating the path component of a URL, 
    including adding nodes, parsing path strings, and building path strings with optional encoding.
    """

    def __init__(self):
        """
        Initializes the UrlPath object with an empty list of segments and a flag indicating 
        the presence of an end tag.
        """
        self.segments = []
        self.with_end_tag = False

    def add(self, segment):
        """
        Adds a segment to the list of segments in the UrlPath.
        :param segment: str, the segment to add.
        """
        self.segments.append(segment)

    def parse(self, path, charset):
        """
        Parses a given path string and populates the list of segments in the UrlPath.
        :param path: str, the path string to parse.
        :param charset: str, the character encoding of the path string.
        """
        parsed_path = urllib.parse.unquote(path)
        self.segments = urllib.parse.urlparse(parsed_path).path.split('/')

    @staticmethod
    def fix_path(path):
        """
        Fixes the given path string by removing leading and trailing slashes.
        :param path: str, the path string to fix.
        :return: str, the fixed path string.
        """
        return '/' + path.strip('/')
    
if __name__ == "__main__":
    url_path = UrlPath()
    url_path.add("home")
    url_path.add("user")
    print(url_path.segments)  # Output: ['home', 'user']
    url_path.parse("/home/user/docs", "utf-8")
    print(url_path.segments)  # Output: ['', 'home', 'user', 'docs']
    fixed_path = UrlPath.fix_path("//home/user//")
    print(fixed_path)  # Output: /home/user