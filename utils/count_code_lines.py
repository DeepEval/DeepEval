def count_code_lines(file_path):
    """
    Count the number of non-empty and non-comment lines in a Python file
    """
    in_comment_block = False
    code_lines = 0
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if in_comment_block:
                if '"""' in line or "'''" in line:
                    in_comment_block = False
                continue
            if line.startswith('"""') or line.startswith("'''"):
                in_comment_block = True
                # Check if the comment block ends on the same line
                if line.count('"""') == 2 or line.count("'''") == 2:
                    in_comment_block = False
                continue
            if not line or line.startswith("#"):
                continue
            code_lines += 1

    return code_lines
