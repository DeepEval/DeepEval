import re


def remove_empty_lines(text):
    lines = text.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    cleaned_text = "\n".join(non_empty_lines)
    return cleaned_text


def extract_text(text):
    """
    extract requirement description from text according to certain rules.
    """
    patterns = [
        (r'""".*?"""', '"""'),
        (r"'''.*?'''", "'''"),
    ]
    for pattern, strip_chars in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            matched_text = match.group(0).strip(strip_chars)
            matched_text = re.sub(r"Examples.*", "", matched_text, flags=re.DOTALL)
            matched_text = re.sub(r"Example.*", "", matched_text, flags=re.DOTALL)
            matched_text = re.sub(r"Examples:.*", "", matched_text, flags=re.DOTALL)
            matched_text = re.sub(r"Example:.*", "", matched_text, flags=re.DOTALL)
            matched_text = re.sub(r"for example:.*", "", matched_text, flags=re.DOTALL)
            matched_text = re.sub(r"for examble:.*", "", matched_text, flags=re.DOTALL)
            matched_text = re.sub(r"For example:.*", "", matched_text, flags=re.DOTALL)
            matched_text = re.sub(r"For example,*", "", matched_text, flags=re.DOTALL)
            matched_text = re.sub(r"example:", "", matched_text, flags=re.DOTALL)
            matched_text = re.sub(r">>>.*", "", matched_text, flags=re.DOTALL)

            matched_text = remove_empty_lines(matched_text) 
            matched_text = matched_text.lstrip()
            matched_text = f'"{matched_text}"'

            return matched_text

