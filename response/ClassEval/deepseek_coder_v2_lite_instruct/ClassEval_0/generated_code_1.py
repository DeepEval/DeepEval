import logging
import datetime
import jwt

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class AccessGatewayFilter:
    """
    This class is a filter used for accessing gateway filtering, primarily for authentication and access log recording.
    """

    def __init__(self):
        self.allowed_prefixes = ["/api", "/login"]

    def filter(self, request):
        """
        Filter the incoming request based on certain rules and conditions.
        :param request: dict, the incoming request details
        :return: bool, True if the request is allowed, False otherwise
        """
        logging.debug(f"Filtering request: {request}")
        if self.is_start_with(request.get('path', '')):
            user = self.get_jwt_user(request)
            if user:
                self.set_current_user_info_and_log(user)
                return True
        return False

    def is_start_with(self, request_uri):
        """
        Check if the request URI starts with certain prefixes.
        Currently, the prefixes being checked are "/api" and "/login".
        :param request_uri: str, the URI of the request
        :return: bool, True if the URI starts with certain prefixes, False otherwise
        """
        logging.debug(f"Checking if URI starts with: {request_uri}")
        return any(request_uri.startswith(prefix) for prefix in self.allowed_prefixes)

    def get_jwt_user(self, request):
        """
        Get the user information from the JWT token in the request.
        :param request: dict, the incoming request details
        :return: dict or None, the user information if the token is valid, None otherwise
        """
        auth_header = request.get('headers', {}).get('Authorization', None)
        if auth_header and 'jwt' in auth_header:
            try:
                decoded_token = jwt.decode(auth_header['jwt'], algorithms=["HS256"])
                logging.debug(f"Decoded token: {decoded_token}")
                return {'user': decoded_token.get('user', {})}
            except jwt.InvalidTokenError:
                logging.debug("Invalid JWT token")
        return None

    def set_current_user_info_and_log(self, user):
        """
        Set the current user information and log the access.
        :param user: dict, the user information
        :return: None
        """
        logging.debug(f"Setting current user info: {user}")
        # Here you can add code to set the current user information in a session or context
        # For example, setting a global variable or updating a database record
        logging.info(f"User accessed the system: {user}")

if __name__ == "__main__":
    # Test cases
    filter = AccessGatewayFilter()

    # Test filter method
    request1 = {'path': '/login', 'method': 'POST'}
    print(f"Test filter for /login: {filter.filter(request1)}")

    # Test is_start_with method
    print(f"Test is_start_with for /api/data: {filter.is_start_with('/api/data')}")
    print(f"Test is_start_with for /user/profile: {filter.is_start_with('/user/profile')}")

    # Test get_jwt_user method
    request2 = {'headers': {'Authorization': {'user': {'name': 'user1'}, 'jwt': 'user1' + str(datetime.date.today())}}}
    user = filter.get_jwt_user(request2)
    print(f"Test get_jwt_user: {user}")

    # Test set_current_user_info_and_log method
    user = {'name': 'user1', 'address': '127.0.0.1'}
    filter.set_current_user_info_and_log(user)