class ArgumentParser:
    """
    This is a class for parsing command line arguments to a dictionary.
    """
    
    def __init__(self):
        """
        Initialize the fields.
        self.arguments is a dict that stores the args in a command line
        self.required is a set that stores the required arguments
        self.types is a dict that stores type of every argument.
        """
        self.arguments = {}
        self.required = set()
        self.types = {}

    def parse_arguments(self, command_string):
        """
        Parses the given command line argument string and stores the parsed result in the arguments dictionary.
        Checks for missing required arguments, if any, and returns False with the missing argument names,
        otherwise returns True.
        """
        import shlex

        # Parse command_string
        tokens = shlex.split(command_string)
        parsed_arguments = {}

        # Process tokens
        for token in tokens:
            if token.startswith('--'):
                key, value = token[2:].split('=', 1)
                parsed_arguments[key] = self._convert_type(key, value)
            elif token.startswith('-'):
                key = token[1:]
                if key in self.types:
                    parsed_arguments[key] = self._convert_type(key, tokens[tokens.index(token) + 1])
                else:
                    parsed_arguments[key] = True

        # Update self.arguments with parsed arguments
        self.arguments.update(parsed_arguments)

        # Check for missing required arguments
        missing_args = self.required - parsed_arguments.keys()
        if missing_args:
            return False, missing_args
        
        return True, None

    def get_argument(self, key):
        """
        Retrieves the value of the specified argument from the arguments dictionary and returns it.
        """
        return self.arguments.get(key)

    def add_argument(self, arg, required=False, arg_type=str):
        """
        Adds an argument to self.types and self.required.
        Check if it is a required argument and store the argument type.
        """
        self.types[arg] = arg_type
        if required:
            self.required.add(arg)

    def _convert_type(self, arg, value):
        """
        Try to convert the type of input value by searching in self.types.
        """
        try:
            return self.types[arg](value)
        except (ValueError, KeyError):
            return value

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('arg1', True, int)
    print(parser.required)
    print(parser.types)

    result, missing = parser.parse_arguments("--arg1=21 -arg2 value2 --option1= True -option2= True")
    print(result)
    print(missing)
    print(parser.arguments)

    print(parser.get_argument('arg2'))

    print(parser._convert_type('arg1', '21'))