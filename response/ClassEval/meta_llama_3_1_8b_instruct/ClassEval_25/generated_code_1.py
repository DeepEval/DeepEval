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
        if 'cookies' in response:
            self.cookies = response['cookies']
            self._save_cookies()
        else:
            print("No cookies found in the response.")

    def load_cookies(self):
        """
        Loads the cookies from the cookies_file to the cookies data.
        :return: The cookies data, dict.
        """
        try:
            with open(self.cookies_file, 'r') as file:
                self.cookies = json.load(file)
                return self.cookies
        except FileNotFoundError:
            print("Cookies file not found.")
            return None

    def _save_cookies(self):
        """
        Saves the cookies to the cookies_file, and returns True if successful, False otherwise.
        :return: True if successful, False otherwise.
        """
        try:
            with open(self.cookies_file, 'w') as file:
                json.dump(self.cookies, file)
                return True
        except Exception as e:
            print(f"Failed to save cookies: {str(e)}")
            return False

if __name__ == "__main__":
    # Test case for get_cookies
    cookies_util = CookiesUtil('cookies.json')
    response = {'cookies': {'key1': 'value1', 'key2': 'value2'}}
    cookies_util.get_cookies(response)
    print(cookies_util.cookies)

    # Test case for load_cookies
    cookies_util = CookiesUtil('cookies.json')
    cookies = cookies_util.load_cookies()
    print(cookies)

    # Test case for _save_cookies
    cookies_util = CookiesUtil('cookies.json')
    cookies_util.cookies = {'key1': 'value1', 'key2': 'value2'}
    output = cookies_util._save_cookies()
    print(output)