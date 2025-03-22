import os
def save_code(code, file_name, save_code_path):
    """
    Save code to a file.
    """
    if not os.path.exists(save_code_path):
        os.makedirs(save_code_path)
    with open(save_code_path + '/' + file_name, "w") as file:
        file.write(code)