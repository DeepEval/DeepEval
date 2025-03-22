import os
import re


def get_task_name(path)->list:
    """
    Get all the folder names under the specified path, and sort them in a natural order
    The goal is to get the specific task names in a batch of experimental results
    """
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    entries = os.listdir(path)
    subfolders = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
    subfolders.sort(key=natural_sort_key)
    return subfolders
