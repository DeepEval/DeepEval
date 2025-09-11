import urllib.parse

class UrlPath:
    """
    The  class is a utility for encapsulating and manipulating the path component of a URL, including adding nodes, parsing path strings, and building path strings with optional encoding.
    """

    def __init__(self):
        """
        Initializes the UrlPath object with an empty list of segments and a flag indicating the presence of an end tag.
        """
        self.segments = []
        self.with_end_tag = False

    def add(self, segment):
        """
        Adds a segment to the list of segments in the UrlPath.
        :param segment: str, the segment to add.
        >>> url_path = UrlPath()
        >>> url_path.add('foo')
        >>> url_path.add('bar')

        url_path.segments = ['foo', 'bar']
        """
        self.segments.append(segment)

    def parse(self, path, charset):
        """
        Parses a given path string and populates the list of segments in the UrlPath.
        :param path: str, the path string to parse.
        :param charset: str, the character encoding of the path string.
        >>> url_path = UrlPath()
        >>> url_path.parse('/foo/bar/', 'utf-8')

        url_path.segments = ['foo', 'bar']
        """
        # Remove leading and trailing slashes
        path = path.strip('/')
        self.segments = urllib.parse.urlparse(path).path.split('/')
        if self.segments[-1] == '':
            self.segments.pop()
        self.with_end_tag = path.endswith('/')

    @staticmethod
    def fix_path(path):
        """
        Fixes the given path string by removing leading and trailing slashes.
        :param path: str, the path string to fix.
        :return: str, the fixed path string.
        >>> url_path = UrlPath()
        >>> url_path.fix_path('/foo/bar/')
        'foo/bar'

        """
        return path.strip('/')

if __name__ == "__main__":
    instance = UrlPath()
    # Test case for add
    instance.add('foo')
    instance.add('bar')
    print(instance.segments)  # Output: ['foo', 'bar']

    # Test case for parse
    instance = UrlPath()
    instance.parse('/foo/bar/', 'utf-8')
    print(instance.segments)  # Output: ['foo', 'bar']
    print(instance.with_end_tag)  # Output: False

    # Test case for fix_path
    print(UrlPath.fix_path('/foo/bar/'))  # Output: foo/bar