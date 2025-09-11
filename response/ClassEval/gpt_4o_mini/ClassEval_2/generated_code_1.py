class ArgumentParser:
    """
    This is a class for parsing command line arguments to a dictionary.
    """

    def __init__(self):
        """
        Initialize the fields.
        self.arguments is a dict that stores the args in a command line
        self.requried is a set that stores the required arguments
        self.types is a dict that stores type of every arguments.
        """
        self.arguments = {}
        self.required = set()
        self.types = {}

    def parse_arguments(self, command_string):
        """
        Parses the given command line argument string and invoke _convert_type to stores the parsed result in specific type in the arguments dictionary.
        Checks for missing required arguments, if any, and returns False with the missing argument names, otherwise returns True.
        :param command_string: str, command line argument string, formatted like "python script.py --arg1=value1 -arg2 value2 --option1 -option2"
        :return tuple: (True, None) if parsing is successful, (False, missing_args) if parsing fails,
            where missing_args is a set of the missing argument names which are str.
        """
        import re
        
        # Regex to match arguments
        arg_pattern = re.compile(r'(--\w+|-\w+)(=([^ ]+))?')

        # Find all matches
        matches = arg_pattern.findall(command_string)

        for match in matches:
            arg = match[0]
            value = match[2] if match[1] else True  # Assign True if no value is provided

            arg_name = arg.lstrip('-')
            self.arguments[arg_name] = self._convert_type(arg_name, value)

        # Check for missing required arguments
        missing_args = self.required - set(self.arguments.keys())
        if missing_args:
            return False, missing_args
        return True, None

    def get_argument(self, key):
        """
        Retrieves the value of the specified argument from the arguments dictionary and returns it.
        :param key: str, argument name
        :return: The value of the argument, or None if the argument does not exist.
        """
        return self.arguments.get(key)

    def add_argument(self, arg, required=False, arg_type=str):
        """
        Adds an argument to self.types and self.required.
        Check if it is a required argument and store the argument type.
        If the argument is set as required, it wull be added to the required set.
        The argument type and name are stored in the types dictionary as key-value pairs.
        :param arg: str, argument name
        :param required: bool, whether the argument is required, default is False
        :param arg_type:str, Argument type, default is str
        """
        if required:
            self.required.add(arg)
        self.types[arg] = arg_type

    def _convert_type(self, arg, value):
        """
        Try to convert the type of input value by searching in self.types.
        :param value: str, the input value in command line
        :return: return corresponding value in self.types if convert successfully, or the input value oherwise
        """
        arg_type = self.types.get(arg, str)
        try:
            return arg_type(value)
        except ValueError:
            return value  # Return the original value if conversion fails

# Test cases
if __name__ == "__main__":
    parser = ArgumentParser()

    # Test add_argument
    parser.add_argument('arg1', True, int)
    parser.add_argument('arg2', False, str)
    print(parser.required)  # Expected: {'arg1'}
    print(parser.types)     # Expected: {'arg1': <class 'int'>, 'arg2': <class 'str'>}

    # Test parse_arguments
    result, missing_args = parser.parse_arguments("python script.py --arg1=21 --arg2=value2 --option1 -option2")
    print(result)          # Expected: True
    print(missing_args)   # Expected: None
    print(parser.arguments) # Expected: {'arg1': 21, 'arg2': 'value2', 'option1': True, 'option2': True}

    # Test get_argument
    print(parser.get_argument('arg2'))  # Expected: 'value2'
    print(parser.get_argument('arg3'))  # Expected: None

    # Test missing required argument
    result, missing_args = parser.parse_arguments("python script.py --arg2=value2 --option1 -option2")
    print(result)          # Expected: False
    print(missing_args)   # Expected: {'arg1'}