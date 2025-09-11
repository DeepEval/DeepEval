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
        :param key:str
        :return:str,the value corresponding to the key
        """
        key = self._convert_key(key)
        return self._data[key]

    def __setitem__(self, key, value):
        """
        Set the value corresponding to the key to the specified value
        :param key:str
        :param value:str, the specified value
        :return:None
        """
        key = self._convert_key(key)
        self._data[key] = value

    def __delitem__(self, key):
        """
        Delete the value corresponding to the key
        :param key:str
        :return:None
        """
        key = self._convert_key(key)
        del self._data[key]

    def __iter__(self):
        """
        Returning Iterable Objects with Own Data
        :return:Iterator
        """
        return iter(self._data)

    def __len__(self):
        """
        Returns the length of the own data
        :return:int, length of data
        """
        return len(self._data)

    def _convert_key(self, key):
        """
        convert key string into camel case
        :param key:str
        :return:str, converted key string
        """
        return self._to_camel_case(key)

    @staticmethod
    def _to_camel_case(key):
        """
        convert key string into camel case
        :param key:str
        :return:str, converted key string
        """
        parts = key.split('_')
        return parts[0] + ''.join(part.capitalize() for part in parts[1:])

# Test cases to validate the functionality of the CamelCaseMap class
if __name__ == "__main__":
    # Test __setitem__ and __getitem__
    camelize_map = CamelCaseMap()
    camelize_map['first_name'] = 'John'
    print(camelize_map['first_name'])  # Output: John

    # Test __setitem__ to update value
    camelize_map['first_name'] = 'Jane'
    print(camelize_map['first_name'])  # Output: Jane

    # Test __delitem__
    del camelize_map['first_name']
    print('first_name' in camelize_map)  # Output: False

    # Test __len__
    camelize_map['first_name'] = 'John'
    camelize_map['last_name'] = 'Doe'
    camelize_map['age'] = 30
    print(len(camelize_map))  # Output: 3

    # Test __iter__
    for key in camelize_map:
        print(key)  # Output: firstName, lastName, age

    # Test _to_camel_case
    print(CamelCaseMap._to_camel_case('first_name'))  # Output: firstName