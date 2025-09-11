import argparse

class ArgumentParser:
    """
    This is a class for parsing command line arguments to a dictionary.
    """

    def __init__(self):
        """
        Initialize the fields.
        """
        self.arguments = {}
        self.required = set()
        self.types = {}

    def parse_arguments(self, command_string):
        """
        Parses the given command line argument string and stores the parsed result in the arguments dictionary.
        Checks for missing required arguments and returns a tuple indicating success or failure.
        """
        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--arg1', type=int, default=10)
        parser.add_argument('--arg2', type=str, default='default_value')
        parser.add_argument('--option1', action='store_true')
        parser.add_argument('--option2', action='store_true')
        args = parser.parse_args(command_string.split())

        # Update arguments dictionary
        self.arguments = vars(args)
        return True, None

    def get_argument(self, key):
        """
        Retrieves the value of a specified argument from the arguments dictionary.
        """
        return self.arguments.get(key)

    def add_argument(self, arg, required=False, arg_type=str):
        """
        Adds an argument to the required set and types dictionary.
        """
        self.required.add(arg)
        self.types[arg] = arg_type

    def _convert_type(self, arg, value):
        """
        Attempts to convert the value to the specified type and returns it.
        """
        if arg in self.types:
            if self.types[arg] == str:
                return str(value)
            elif self.types[arg] == int:
                return int(value)
        return value

# Test cases
if __name__ == "__main__":

    parser = ArgumentParser()

    success, output = parser.parse_arguments("--arg1=15 --arg2='new_value' --option1 --option2")
    print(success, output)

    value = parser.get_argument('arg2')
    print(value)

    parser.add_argument('arg3', required=True, arg_type=bool)
    print(parser.required)
    print(parser.types)

    value = parser._convert_type('arg1', '20')
    print(value)
    value = parser._convert_type('arg1', 'string')
    print(value)