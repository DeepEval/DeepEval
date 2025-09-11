import json
import os

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
        >>> cookies_util = CookiesUtil('cookies.json')
        >>> cookies_util.get_cookies({'cookies': {'key1': 'value1', 'key2': 'value2'}})
        >>> cookies_util.cookies
        {'key1': 'value1', 'key2': 'value2'}
        """
        self.cookies = response.get('cookies', {})
        self._save_cookies()

    def load_cookies(self):
        """
        Loads the cookies from the cookies_file to the cookies data.
        :return: The cookies data, dict.
        >>> cookies_util = CookiesUtil('cookies.json')
        >>> cookies_util.load_cookies()
        {'key1': 'value1', 'key2': 'value2'}
        """
        if os.path.exists(self.cookies_file):
            with open(self.cookies_file, 'r') as f:
                self.cookies = json.load(f)
        else:
            self.cookies = {}
        return self.cookies

    def _save_cookies(self):
        """
        Saves the cookies to the cookies_file, and returns True if successful, False otherwise.
        :return: True if successful, False otherwise.
        >>> cookies_util = CookiesUtil('cookies.json')
        >>> cookies_util.cookies = {'key1': 'value1', 'key2': 'value2'}
        >>> cookies_util._save_cookies()
        True
        """
        try:
            with open(self.cookies_file, 'w') as f:
                json.dump(self.cookies, f)
            return True
        except Exception as e:
            print(f"An error occurred while saving cookies: {e}")
            return False

# Test cases
if __name__ == "__main__":
    # Test get_cookies
    cookies_util = CookiesUtil('cookies.json')
    cookies_util.get_cookies({'cookies': {'key1': 'value1', 'key2': 'value2'}})
    print("Cookies after get_cookies:", cookies_util.cookies)

    # Test load_cookies
    loaded_cookies = cookies_util.load_cookies()
    print("Loaded Cookies:", loaded_cookies)

    # Test _save_cookies
    save_result = cookies_util._save_cookies()
    print("Cookies saved successfully:", save_result)

    # Re-load to verify
    loaded_cookies_again = cookies_util.load_cookies()
    print("Re-loaded Cookies:", loaded_cookies_again)