def method():
    # input_string = input("Enter a space-delimited string of numbers from 'zero' to 'nine': ")

    # 修改为固定值
    input_string = "zero one two three four five six seven eight nine"
    
    numbers = input_string.split()
    
    # Convert the string numbers to integers
    numbers = [int(num) for num in numbers if num in ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']]
    
    # Sort the numbers
    numbers.sort()

    # Convert the sorted numbers back to strings
    output_string = ' '.join([str(num) for num in numbers]) 

    return output_string

# Test case
output = method()
print(f"Sorted output: {output}")