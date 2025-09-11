import re

class ArgumentParser:
    def __init__(self):
        self.arguments = {}
        self.required = set()
        self.types = {}

    def parse_arguments(self, command_string):
        args = command_string.split()
        for arg in args:
            if not arg.startswith('--'):
                continue
            key, value = arg.split('=', 1)
            self.arguments[key] = value
            self.required.add(key) if value.lower() == 'true' else self.required.remove(key)
            self.types[key] = str
        for arg in self.required:
            if arg not in self.arguments:
                return False, f"Missing required argument {arg}"
        return True, None

    def get_argument(self, key):
        return self.arguments[key] if key in self.arguments else None

    def add_argument(self, arg, required=False, arg_type=str):
        self.required.add(arg) if required else self.required.remove(arg)
        self.types[arg] = arg_type

    def _convert_type(self, arg, value):
        arg_type = self.types[arg]
        if arg_type == str:
            return value
        elif arg_type == int:
            return int(value)
        elif arg_type == float:
            return float(value)
        else:
            return value

if __name__ == "__main__":
    parser = ArgumentParser()
    args = "python script.py --arg1=value1 -arg2 value2 --option1 -option2"
    status, missing_args = parser.parse_arguments(args)
    if status:
        print("Parsing successful!")
    else:
        print("Missing required arguments:", missing_args)
    print(parser.arguments)