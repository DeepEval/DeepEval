import logging
import datetime

class AccessGatewayFilter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def filter(self, request):
        if self.is_start_with(request['path']):
            self.logger.info(f"Path {request['path']} is allowed")
            return True
        else:
            self.logger.warning(f"Path {request['path']} is denied")
            return False
    
    def is_start_with(self, request_uri):
        return request_uri.startswith(('/api', '/login'))
    
    def get_jwt_user(self, request):
        try:
            auth_header = request['headers']['Authorization']
            jwt = auth_header['jwt']
            user_info = auth_header['user']
            if user_info is None:
                return None
            else:
                return user_info
        except:
            return None
    
    def set_current_user_info_and_log(self, user):
        self.logger.info(f"User {user['name']} logged in")
        self.logger.info(f"IP address {user['address']} logged in")
        return user

if __name__ == "__main__":
    filter = AccessGatewayFilter()
    request = {'path': '/login', 'method': 'POST', 'headers': {'Authorization': {'user': {'name': 'user1'}, 'jwt': 'user1'+str(datetime.date.today())}}}
    print(filter.filter(request))
    user = {'name': 'user1', 'address': '127.0.0.1'}
    print(filter.set_current_user_info_and_log(user))

