class CamelCaseMap:
    """
    This is a custom class that allows keys to be in camel case style by converting them from underscore style, which provides dictionary-like functionality.
    """

    def __init__(self):
        """
        Initialize data to an empty dictionary
        """
        self._data = {}

    def __getitem__(self, key):
        """
        Return the value corresponding to the key
        :param key: str
        :return: str, the value corresponding to the key
        """
        camel_key = self._convert_key(key)
        return self._data[camel_key]

    def __setitem__(self, key, value):
        """
        Set the value corresponding to the key to the specified value
        :param key: str
        :param value: str, the specified value
        :return: None
        """
        camel_key = self._convert_key(key)
        self._data[camel_key] = value

    def __delitem__(self, key):
        """
        Delete the value corresponding to the key
        :param key: str
        :return: None
        """
        camel_key = self._convert_key(key)
        del self._data[camel_key]

    def __iter__(self):
        """
        Returning Iterateable Objects with Own Data
        :return: Iterator
        """
        return iter(self._data)

    def __len__(self):
        """
        Returns the length of the own data
        :return: int, length of data
        """
        return len(self._data)

    def _convert_key(self, key):
        """
        convert key string into camel case
        :param key: str
        :return: str, converted key string
        """
        return self._to_camel_case(key)

    @staticmethod
    def _to_camel_case(key):
        """
        convert key string into camel case
        :param key: str
        :return: str, converted key string
        """
        parts = key.split('_')
        return parts[0] + ''.join(word.capitalize() for word in parts[1:])

# Test cases
if __name__ == "__main__":
    camelize_map = CamelCaseMap()
    
    # Test __setitem__
    camelize_map['first_name'] = 'John'
    print(camelize_map['first_name'])  # Output: John

    # Test __setitem__ and __getitem__
    camelize_map['last_name'] = 'Doe'
    print(camelize_map['last_name'])  # Output: Doe

    # Test __delitem__
    del camelize_map['last_name']
    try:
        print(camelize_map['last_name'])  # Should raise KeyError
    except KeyError:
        print("KeyError: 'last_name' not found")

    # Test __iter__
    camelize_map['age'] = 30
    for key in camelize_map:
        print(key)  # Output: firstName, age

    # Test __len__
    print(len(camelize_map))  # Output: 2