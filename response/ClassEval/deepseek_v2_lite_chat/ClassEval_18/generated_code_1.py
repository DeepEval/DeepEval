class CamelCaseMap:
    """
    This is a custom class that allows keys to be in camel case style by converting them from underscore style,
    which provides dictionary-like functionality.
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
        # Convert the key to camel case
        camel_key = self._convert_key(key)
        return self._data.get(camel_key, "Key not found")

    def __setitem__(self, key, value):
        """
        Set the value corresponding to the key to the specified value
        :param key: str
        :param value: str, the specified value
        :return: None
        """
        # Convert the key to camel case
        camel_key = self._convert_key(key)
        self._data[camel_key] = value

    def __delitem__(self, key):
        """
        Delete the value corresponding to the key
        :param key: str
        :return: None
        """
        # Convert the key to camel case
        camel_key = self._convert_key(key)
        if camel_key in self._data:
            del self._data[camel_key]

    def __iter__(self):
        """
        Returning Iterateable Objects with Own Data
        :return: Iterator
        """
        return _dict_keyiterator(self._data)

    def __len__(self):
        """
        Returns the length of the own data
        :return: int, length of data
        """
        return len(self._data)

    def _convert_key(self, key):
        """
        Convert key string into camel case
        :param key: str
        :return: str, converted key string
        """
        # Convert underscore-separated words to camel case
        return ''.join(word.title() for word in key.split('_'))

    @staticmethod
    def _to_camel_case(key):
        """
        Convert key string into camel case
        :param key: str
        :return: str, converted key string
        """
        # Convert underscore-separated words to camel case
        return ''.join(word.title() for word in key.split('_'))


class _dict_keyiterator:
    """Iterator for a dictionary's keys."""

    def __init__(self, d):
        self.d = d
        self.keys = list(d.keys())
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.keys):
            raise StopIteration
        result = self.keys[self.index]
        self.index += 1
        return result


# Test cases
if __name__ == "__main__":
    camelize_map = CamelCaseMap()
    camelize_map['first_name'] = 'John'
    camelize_map['last_name'] = 'Doe'
    camelize_map['age'] = 30
    print("John", camelize_map['first_name'])  # Test __getitem__
    camelize_map['first_name'] = 'New Name'
    print(camelize_map['first_name'])  # Test __setitem__
    del camelize_map['last_name']
    print(camelize_map['last_name'])  # Test __delitem__
    print(len(camelize_map))  # Test __len__
    for key in camelize_map:
        print(key)  # Test __iter__