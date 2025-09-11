import argparse
import sys

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
        parser = argparse.ArgumentParser()
        for arg in self.types:
            if self.types[arg] == bool:
                parser.add_argument(f"--{arg}", action="store_true")
            else:
                parser.add_argument(f"--{arg}", type=self.types[arg])
        try:
            args = parser.parse_args(command_string.split()[2:])
            for key in args.__dict__:
                self.arguments[key] = self._convert_type(key, args.__dict__[key])
            for arg in self.required:
                if arg not in self.arguments:
                    return False, {arg}
            return True, None
        except SystemExit:
            return False, self.required

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
        self.types[arg] = arg_type
        if required:
            self.required.add(arg)

    def _convert_type(self, arg, value):
        """
        Try to convert the type of input value by searching in self.types.
        :param value: str, the input value in command line
        :return: return corresponding value in self.types if convert successfully, or the input value oherwise
        """
        try:
            return self.types[arg](value)
        except ValueError:
            return value

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('arg1', True, int)
    parser.add_argument('arg2', False, float)
    parser.add_argument('option1', False, bool)
    
    # Test case for parse_arguments
    command_string = "script.py --arg1=21 --arg2=22 --option1"
    output = parser.parse_arguments(command_string)
    print(output)  # Output: (True, None)

    # Test case for get_argument
    parser.parse_arguments(command_string)
    output = parser.get_argument('arg1')
    print(output)  # Output: 21

    # Test case for add_argument
    parser = ArgumentParser()
    parser.add_argument('arg1', True, int)
    output = parser.required
    print(output)  # Output: {'arg1'}

    # Test case for _convert_type
    output = parser._convert_type('arg1', '21')
    print(output)  # Output: 21

    # Test case for parse_arguments with missing arguments
    command_string = "script.py --arg2=22 --option1"
    output = parser.parse_arguments(command_string)
    print(output)  # Output: (False, {'arg1'})