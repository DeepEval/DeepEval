def method():
  
    my_dict = {"name": "John", "Age": 30, "City": "New York"}  

    if len(my_dict) == 0:
      return False
    
    keys_lower = [key.lower() for key in my_dict.keys()]
    keys_upper = [key.upper() for key in my_dict.keys()]
    return keys_lower == list(my_dict.keys()) or keys_upper == list(my_dict.keys())

output = method()
print(output)