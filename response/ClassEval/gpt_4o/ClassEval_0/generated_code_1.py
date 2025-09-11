import logging
import datetime
import unittest

class AccessGatewayFilter:
    """
    This class is a filter used for accessing gateway filtering, primarily for authentication and access log recording.
    """

    def __init__(self):
        # Set up logging configuration
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def filter(self, request):
        """
        Filter the incoming request based on certain rules and conditions.
        :param request: dict, the incoming request details
        :return: bool, True if the request is allowed, False otherwise
        """
        if self.is_start_with(request.get('path', '')):
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
        prefixes = ["/api", "/login"]
        return any(request_uri.startswith(prefix) for prefix in prefixes)

    def get_jwt_user(self, request):
        """
        Get the user information from the JWT token in the request.
        :param request: dict, the incoming request details
        :return: dict or None, the user information if the token is valid, None otherwise
        """
        headers = request.get('headers', {})
        authorization = headers.get('Authorization', {})
        jwt = authorization.get('jwt', '')

        # Mocked token validation
        if jwt.endswith(str(datetime.date.today())):
            return authorization.get('user', None)
        return None

    def set_current_user_info_and_log(self, user):
        """
        Set the current user information and log the access.
        :param user: dict, the user information
        :return: None
        """
        self.logger.info(f"User {user.get('name')} accessed the system from {user.get('address', 'Unknown')}")


class TestAccessGatewayFilter(unittest.TestCase):

    def test_filter(self):
        filter = AccessGatewayFilter()
        request = {
            'path': '/login',
            'method': 'POST',
            'headers': {
                'Authorization': {
                    'user': {'name': 'user1', 'address': '127.0.0.1'},
                    'jwt': 'user1' + str(datetime.date.today())
                }
            }
        }
        self.assertTrue(filter.filter(request))

    def test_is_start_with(self):
        filter = AccessGatewayFilter()
        self.assertTrue(filter.is_start_with('/api/data'))
        self.assertTrue(filter.is_start_with('/login'))
        self.assertFalse(filter.is_start_with('/home'))

    def test_get_jwt_user(self):
        filter = AccessGatewayFilter()
        request = {
            'headers': {
                'Authorization': {
                    'user': {'name': 'user1'},
                    'jwt': 'user1' + str(datetime.date.today())
                }
            }
        }
        self.assertEqual(filter.get_jwt_user(request), {'name': 'user1'})

    def test_set_current_user_info_and_log(self):
        filter = AccessGatewayFilter()
        user = {'name': 'user1', 'address': '127.0.0.1'}
        # This won't assert anything, just a smoke test to ensure no exceptions
        filter.set_current_user_info_and_log(user)


if __name__ == "__main__":
    unittest.main()