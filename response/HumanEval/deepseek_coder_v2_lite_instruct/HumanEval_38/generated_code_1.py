def method(input_string):
    # Initialize an empty list to store the cycled groups
    cycled_groups = []
    
    # Iterate over the input string in steps of 3
    for i in range(0, len(input_string), 3):
        # Get the current group of 3 characters
        group = input_string[i:i+3]
        
        # If the group has 3 characters, cycle them
        if len(group) == 3:
            cycled_group = group[1:] + group[0]
            cycled_groups.append(cycled_group)
        else:
            # If the group has less than 3 characters, just append it as is
            cycled_groups.append(group)
    
    # Join the cycled groups back into a single string
    output = ''.join(cycled_groups)
    
    return output

# Test case
input_string = "abcdefghi"
output = method(input_string)
print(output)  # Expected output: "bcaefdghi"