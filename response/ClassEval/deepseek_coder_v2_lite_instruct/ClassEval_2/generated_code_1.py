class ArgumentParser:
    """
    This is a class for parsing command line arguments to a dictionary.
    """

    def __init__(self):
        """
        Initialize the fields.
        self.arguments is a dict that stores the args in a command line
        self.required is a set that stores the required arguments
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
        args = re.findall(r'[\-\-]?(\w+)(?:\s*=\s*([^\s]+))?', command_string)
        missing_args = set()
        
        for arg, value in args:
            if arg in self.required and arg not in self.arguments:
                missing_args.add(arg)
            else:
                self.arguments[arg] = value if value else True
        
        for arg, arg_type in self.types.items():
            if arg in self.arguments:
                self.arguments[arg] = self._convert_type(arg, self.arguments[arg])
        
        if missing_args:
            return (False, missing_args)
        return (True, None)

    def get_argument(self, key):
        """
        Retrieves the value of the specified argument from the arguments dictionary and returns it.
        :param key: str, argument name
        :return: The value of the argument, or None if the argument does not exist.
        """
        return self.arguments.get(key, None)

    def add_argument(self, arg, required=False, arg_type=str):
        """
        Adds an argument to self.types and self.required.
        Check if it is a required argument and store the argument type.
        If the argument is set as required, it will be added to the required set.
        The argument type and name are stored in the types dictionary as key-value pairs.
        :param arg: str, argument name
        :param required: bool, whether the argument is required, default is False
        :param arg_type: str, Argument type, default is str
        """
        self.types[arg] = arg_type
        if required:
            self.required.add(arg)

    def _convert_type(self, arg, value):
        """
        Try to convert the type of input value by searching in self.types.
        :param value: str, the input value in command line
        :return: return corresponding value in self.types if convert successfully, or the input value otherwise
        """
        if arg in self.types:
            if self.types[arg] == int:
                return int(value)
            elif self.types[arg] == float:
                return float(value)
            elif self.types[arg] == bool:
                return value.lower() in ('true', '1')
            else:
                return self.types[arg](value)
        return value

# Test cases
if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Test add_argument
    parser.add_argument('arg1', True, 'int')
    parser.add_argument('arg2', False, 'str')
    parser.add_argument('arg3', False, 'float')
    parser.add_argument('arg4', False, 'bool')
    
    # Test parse_arguments
    result = parser.parse_arguments("python script.py --arg1=21 -arg2 value2 --arg3=3.14 --arg4=true")
    print("Parse Arguments Result:", result)
    print("Arguments:", parser.arguments)
    
    # Test get_argument
    print("Get Argument 'arg1':", parser.get_argument('arg1'))
    print("Get Argument 'arg2':", parser.get_argument('arg2'))
    print("Get Argument 'arg3':", parser.get_argument('arg3'))
    print("Get Argument 'arg4':", parser.get_argument('arg4'))