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
        Gets the cookies from the specified response, and save it to cookies_file.
        :param response: The response to get cookies from, dict.
        """
        self.cookies = response['cookies']
        self._save_cookies()

    def load_cookies(self):
        """
        Loads the cookies from the cookies_file to the cookies data.
        :return: The cookies data, dict.
        """
        if not self.cookies:
            self.cookies = json.load(open(self.cookies_file))
        return self.cookies

    def _save_cookies(self):
        """
        Saves the cookies to the cookies_file, and returns True if successful, False otherwise.
        :return: True if successful, False otherwise.
        """
        try:
            with open(self.cookies_file, 'w') as f:
                json.dump(self.cookies, f)
            return True
        except Exception as e:
            print(f"Error saving cookies: {e}")
            return False

if __name__ == "__main__":
    cookies_file = 'cookies.json'
    cookies_util = CookiesUtil(cookies_file)

    # Test case for get_cookies
    response = {'cookies': {'key1': 'value1', 'key2': 'value2'}}
    cookies_util.get_cookies(response)
    assert cookies_util.cookies == response['cookies']

    # Test case for load_cookies
    cookies_util.load_cookies()
    assert cookies_util.cookies == response['cookies']

    # Test case for _save_cookies
    cookies_util._save_cookies()
    assert json.load(open(cookies_file)) == response['cookies']