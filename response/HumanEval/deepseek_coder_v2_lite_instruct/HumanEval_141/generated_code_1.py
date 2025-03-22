def method():
    import re
    
    def is_valid_filename(filename):
        # Check if the filename contains exactly one dot
        if filename.count('.') != 1:
            return 'No'
        
        # Split the filename into name and extension
        name, extension = filename.rsplit('.', 1)
        
        # Check if the substring before the dot is not empty and starts with a letter
        if not name or not re.match(r'^[a-zA-Z]', name):
            return 'No'
        
        # Check if the substring after the dot is one of the allowed extensions
        allowed_extensions = ['txt', 'exe', 'dll']
        if extension not in allowed_extensions:
            return 'No'
        
        # Check if the name contains more than three digits
        if len(re.findall(r'\d', name)) > 3:
            return 'No'
        
        return 'Yes'
    
    # Example test case
    filename = "example.txt"
    output = is_valid_filename(filename)
    return output

# Run the method and print the result
print(method())