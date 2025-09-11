class CamelCaseMap:
    def __init__(self):
        self._data = {}

    def __getitem__(self, key):
        return self._data[self._convert_key(key)]

    def __setitem__(self, key, value):
        self._data[self._convert_key(key)] = value

    def __delitem__(self, key):
        del self._data[self._convert_key(key)]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def _convert_key(self, key):
        if not isinstance(key, str):
            raise ValueError("Invalid key type")
        return ''.join(x.title() for x in key.split('_'))

    @staticmethod
    def _to_camel_case(key):
        return ''.join(x.title() for x in key.split('_'))

if __name__ == "__main__":
    camelize_map = CamelCaseMap()
    camelize_map['first_name'] = 'John'
    print(camelize_map['first_name'])  # Output: John
    camelize_map['last_name'] = 'Doe'
    print(camelize_map['last_name'])  # Output: Doe
    camelize_map['age'] = 30
    print(camelize_map['age'])  # Output: 30
    print(len(camelize_map))  # Output: 3
    print(camelize_map.keys())  # Output: dict_keys(['first_name', 'last_name', 'age'])
    del camelize_map['last_name']
    print(camelize_map.keys())  # Output: dict_keys(['first_name', 'age'])