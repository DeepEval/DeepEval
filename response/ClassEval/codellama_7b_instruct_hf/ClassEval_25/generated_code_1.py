import json

class CookiesUtil:
    def __init__(self, cookies_file):
        self.cookies_file = cookies_file
        self.cookies = None

    def get_cookies(self, response):
        self.cookies = response.get("cookies")
        with open(self.cookies_file, "w") as f:
            json.dump(self.cookies, f)
        return self.cookies

    def load_cookies(self):
        with open(self.cookies_file, "r") as f:
            self.cookies = json.load(f)
        return self.cookies

    def _save_cookies(self):
        with open(self.cookies_file, "w") as f:
            json.dump(self.cookies, f)
        return True
    
if __name__ == "__main__":
    # Example usage
    cookies_util = CookiesUtil("cookies.json")
    response = {
        "cookies": {
            "session_id": "123456789",
            "user_id": "user_123"
        }
    }
    cookies_util.get_cookies(response)
    loaded_cookies = cookies_util.load_cookies()
    print(loaded_cookies)  # Output: {'session_id': '123456789', 'user_id': 'user_123'}
    cookies_util._save_cookies()
    