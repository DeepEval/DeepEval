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
        :return:str, the value corresponding to the key
        """
        camel_case_key = self._convert_key(key)
        return self._data[camel_case_key]

    def __setitem__(self, key, value):
        """
        Set the value corresponding to the key to the specified value
        :param key:str
        :param value:str, the specified value
        """
        camel_case_key = self._convert_key(key)
        self._data[camel_case_key] = value

    def __delitem__(self, key):
        """
        Delete the value corresponding to the key
        :param key:str
        """
        camel_case_key = self._convert_key(key)
        del self._data[camel_case_key]

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
        return parts[0] + ''.join(word.capitalize() for word in parts[1:])


if __name__ == "__main__":
    # Test case for __getitem__ and __setitem__
    camelize_map = CamelCaseMap()
    camelize_map['first_name'] = 'John'
    output = camelize_map['first_name']
    print(output)  # Should print 'John'

    # Test case for __setitem__
    camelize_map['first_name'] = 'new name'
    output = camelize_map['first_name']
    print(output)  # Should print 'new name'

    # Test case for __delitem__
    camelize_map.__delitem__('first_name')
    output = 'first_name' in camelize_map
    print(output)  # Should print False

    # Test case for __iter__
    camelize_map['first_name'] = 'John'
    camelize_map['last_name'] = 'Doe'
    camelize_map['age'] = 30
    for key in camelize_map:
        print(key)  # Should print camel case keys: firstName, lastName, age

    # Test case for __len__
    output = camelize_map.__len__()
    print(output)  # Should print 3

    # Test case for _convert_key and _to_camel_case
    output = camelize_map._convert_key('first_name')
    print(output)  # Should print 'firstName'