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
            print(f"The file {self.cookies_file} does not exist.")
            return {}
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {self.cookies_file}.")
            return {}

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
            print(f"Failed to save cookies: {e}")
            return False

if __name__ == "__main__":
    cookies_util = CookiesUtil('cookies.json')
    
    # Test load_cookies
    print("Loading cookies:", cookies_util.load_cookies())
    
    # Test get_cookies
    cookies_util.get_cookies({'cookies': {'key1': 'value1', 'key2': 'value2'}})
    print("Loaded cookies:", cookies_util.cookies)
    
    # Test _save_cookies
    print("Save cookies:", cookies_util._save_cookies())
    
    # Test load_cookies after saving
    print("Loading cookies after saving:", cookies_util.load_cookies())