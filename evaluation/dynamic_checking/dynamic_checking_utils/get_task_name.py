import os
import re


def get_task_name(path)->list:
    """
    Get the names of all folders in the specified path and sort them in natural order.
    The goal is to obtain the specific task names from a batch of experimental results.
    """
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    entries = os.listdir(path)
    # Filter out non-directory entries, keeping only folders
    subfolders = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
    subfolders.sort(key=natural_sort_key)
    return subfolders
