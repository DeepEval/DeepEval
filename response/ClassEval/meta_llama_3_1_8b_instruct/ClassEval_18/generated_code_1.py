from typing import Iterator

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
        >>> camelize_map = CamelCaseMap()
        >>> camelize_map['first_name'] = 'John'
        >>> camelize_map.__getitem__('first_name')
        'John'
        """
        camel_case_key = self._to_camel_case(key)
        return self._data.get(camel_case_key)

    def __setitem__(self, key, value):
        """
        Set the value corresponding to the key to the specified value
        :param key:str
        :param value:str, the specified value
        :return:None
        >>> camelize_map = CamelCaseMap()
        >>> camelize_map['first_name'] = 'John'
        >>> camelize_map.__setitem__('first_name', 'new name')
        camelize_map['first_name'] = 'new name'
        """
        camel_case_key = self._to_camel_case(key)
        self._data[camel_case_key] = value

    def __delitem__(self, key):
        """
        Delete the value corresponding to the key
        :param key:str
        :return:None
        >>> camelize_map = CamelCaseMap()
        >>> camelize_map['first_name'] = 'John'
        >>> camelize_map.__delitem__('first_name')
        >>> flag = 'first_name' in camelize_map
        flag = False
        """
        camel_case_key = self._to_camel_case(key)
        if camel_case_key in self._data:
            del self._data[camel_case_key]

    def __iter__(self):
        """
        Returning Iterateable Objects with Own Data
        :return:Iterator
        >>> camelize_map = CamelCaseMap()
        >>> camelize_map['first_name'] = 'John'
        >>> camelize_map['last_name'] = 'Doe'
        >>> camelize_map['age'] = 30
        >>> camelize_map.__iter__()
        <dict_keyiterator object at 0x0000026739977C20>
        """
        return iter(self._data)

    def __len__(self):
        """
        Returns the length of the own data
        :return:int, length of data
        >>> camelize_map = CamelCaseMap()
        >>> camelize_map['first_name'] = 'John'
        >>> camelize_map['last_name'] = 'Doe'
        >>> camelize_map['age'] = 30
        >>> camelize_map.__len__()
        3
        """
        return len(self._data)

    def _convert_key(self, key):
        """
        convert key string into camel case
        :param key:str
        :return:str, converted key string
        >>> camelize_map = CamelCaseMap()
        >>> camelize_map._convert_key('first_name')
        'firstName'
        """
        words = key.split('_')
        return ''.join(word.capitalize() for word in words)

    @staticmethod
    def _to_camel_case(key):
        """
        convert key string into camel case
        :param key:str
        :return:str, converted key string
        >>> camelize_map = CamelCaseMap()
        >>> camelize_map._to_camel_case('first_name')
        'firstName'
        """
        return CamelCaseMap._convert_key(key)


if __name__ == "__main__":
    camelize_map = CamelCaseMap()

    # Test case for __getitem__
    camelize_map['first_name'] = 'John'
    print(camelize_map.__getitem__('first_name'))  # Output: John

    # Test case for __setitem__
    camelize_map.__setitem__('first_name', 'new name')
    print(camelize_map.__getitem__('first_name'))  # Output: new name

    # Test case for __delitem__
    camelize_map.__delitem__('first_name')
    print('first_name' in camelize_map)  # Output: False

    # Test case for __iter__
    camelize_map['last_name'] = 'Doe'
    camelize_map['age'] = 30
    for key in camelize_map:
        print(key)  # Output: lastName, age

    # Test case for __len__
    print(camelize_map.__len__())  # Output: 2

    # Test case for _convert_key
    print(camelize_map._convert_key('first_name'))  # Output: firstName

    # Test case for _to_camel_case
    print(camelize_map._to_camel_case('first_name'))  # Output: firstName