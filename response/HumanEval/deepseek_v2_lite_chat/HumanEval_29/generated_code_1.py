def method(input_list, prefix):
    # Filter the input list using a list comprehension
    output = [s for s in input_list if s.startswith(prefix)]
    return output