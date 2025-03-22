def method(input_dict):
    a = input_dict.get("a", "default_value")
    b = input_dict.get("b", "default_value")
    c = input_dict.get("c", "default_value")
    output = a + b + c
    return output

# Test the method
input_dict = {"a": 1, "b": 2, "c": 3}
print(method(input_dict))  # should print 6

input_dict = {"a": 1, "b": 2}
print(method(input_dict))  # should print 3

input_dict = {"a": 1}
print(method(input_dict))  # should print 2

input_dict = {}
print(method(input_dict))  # should print 0