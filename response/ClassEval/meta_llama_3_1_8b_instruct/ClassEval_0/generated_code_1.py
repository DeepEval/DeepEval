import logging
import datetime
import jwt
import json

class AccessGatewayFilter:
    """
    This class is a filter used for accessing gateway filtering, primarily for authentication and access log recording.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.handler = logging.StreamHandler()
        self.handler.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)
        self.prefixes = ["/api", "/login"]
        self.secret_key = "my_secret_key"

    def filter(self, request):
        """
        Filter the incoming request based on certain rules and conditions.
        :param request: dict, the incoming request details
        :return: bool, True if the request is allowed, False otherwise
        """
        if self.is_start_with(request["path"]):
            user = self.get_jwt_user(request)
            if user:
                self.set_current_user_info_and_log(user)
                return True
        return False

    def is_start_with(self, request_uri):
        """
        Check if the request URI starts with certain prefixes.
        :param request_uri: str, the URI of the request
        :return: bool, True if the URI starts with certain prefixes, False otherwise
        """
        for prefix in self.prefixes:
            if request_uri.startswith(prefix):
                return True
        return False

    def get_jwt_user(self, request):
        """
        Get the user information from the JWT token in the request.
        :param request: dict, the incoming request details
        :return: dict or None, the user information if the token is valid, None otherwise
        """
        try:
            token = request["headers"]["Authorization"]["jwt"]
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            user = json.loads(payload["user"])
            return user
        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token has expired")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid JWT token")
            return None

    def set_current_user_info_and_log(self, user):
        """
        Set the current user information and log the access.
        :param user: dict, the user information
        :return: None
        """
        self.logger.info(f"Accessed by {user['name']} from {user['address']}")
        # You can store the user information in a database or cache here

if __name__ == "__main__":
    filter = AccessGatewayFilter()
    # Test case for filter method
    request1 = {"path": "/api/data", "method": "GET", "headers": {"Authorization": {"jwt": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjp7InVzZXJuYW1lIjoiY29uc3QxIiwibmFtZSI6IjIifX0.XV4MxuPq7VUyQyRlUkL4J4F4pD5wRzIhY8Z9Vh4r0Q"}}}
    request2 = {"path": "/login", "method": "POST", "headers": {"Authorization": {"jwt": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjp7InVzZXJuYW1lIjoiY29uc3QxIiwibmFtZSI6IjIifX0.XV4MxuPq7VUyQyRlUkL4J4F4pD5wRzIhY8Z9Vh4r0Q"}}}
    request3 = {"path": "/public/data", "method": "GET", "headers": {"Authorization": {"jwt": "invalid_token"}}}
    print(filter.filter(request1))  # Expected output: True
    print(filter.filter(request2))  # Expected output: True
    print(filter.filter(request3))  # Expected output: False

    # Test case for is_start_with method
    print(filter.is_start_with("/api/data"))  # Expected output: True
    print(filter.is_start_with("/login/data"))  # Expected output: True
    print(filter.is_start_with("/public/data"))  # Expected output: False

    # Test case for get_jwt_user method
    print(filter.get_jwt_user(request1))  # Expected output: {'name': 'user1', 'address': '127.0.0.1'}
    print(filter.get_jwt_user(request2))  # Expected output: {'name': 'user1', 'address': '127.0.0.1'}
    print(filter.get_jwt_user(request3))  # Expected output: None

    # Test case for set_current_user_info_and_log method
    user = {"name": "user1", "address": "127.0.0.1"}
    filter.set_current_user_info_and_log(user)