def method():
    input_string = "((a))((b))((c))" 
    output = []
    current_group = ""
    open_count = 0
    for char in input_string:
        if char == '(':
            open_count += 1
            current_group += char
        elif char == ')':
            if open_count > 0:
                open_count -= 1
                current_group += char
                if open_count == 0:
                    output.append(current_group)
                    current_group = ""
    return output

output = method()
print(output)