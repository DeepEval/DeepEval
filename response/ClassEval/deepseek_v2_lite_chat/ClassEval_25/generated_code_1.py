import json

class CookiesUtil:
    """
    This is a class as utility for managing and manipulating Cookies, including methods for retrieving, saving, and setting Cookies data.
    """

    def __init__(self, cookies_file):
        """
        Initializes the CookiesUtil with the specified cookies file.
        :param cookies_file: The cookies file to use, str.
        """
        self.cookies_file = cookies_file
        self.cookies = None

    def get_cookies(self, response):
        """
        Gets the cookies from the specified response, and saves it to cookies_file.
        :param response: The response to get cookies from, dict.
        """
        self.cookies = response.cookies
        self._save_cookies()
        return self.cookies

    def load_cookies(self):
        """
        Loads the cookies from the cookies_file to the cookies data.
        :return: The cookies data, dict.
        """
        with open(self.cookies_file, 'r') as f:
            self.cookies = json.load(f)
        return self.cookies

    def _save_cookies(self):
        """
        Saves the cookies to the cookies_file, and returns True if successful, False otherwise.
        :return: True if successful, False otherwise.
        """
        with open(self.cookies_file, 'w') as f:
            json.dump(self.cookies, f)
            return True

# Test cases
if __name__ == "__main__":
    cookies_util = CookiesUtil('cookies.json')
    response = {'cookies': {'key1': 'value1', 'key2': 'value2'}}

    # Get cookies from a response
    print(cookies_util.get_cookies(response))

    # Load cookies
    print(cookies_util.load_cookies())

    # Set cookies
    cookies_util.cookies = {'key1': 'value1', 'key2': 'value2'}
    print(cookies_util._save_cookies())

    # Get cookies after setting
    print(cookies_util.get_cookies(None))