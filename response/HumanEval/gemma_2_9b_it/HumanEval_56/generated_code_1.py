def method():
  # brackets = input("Enter a string of '<' and '>': ")
  
  # 修改为固定值
  brackets = "<>"

  stack = []
  for char in brackets:
    if char == "<":
      stack.append(char)
    elif char == ">":
      if not stack:
        return False
      stack.pop()
  return not stack 

output = method()

if output:
  print("The brackets are balanced.")
else:
  print("The brackets are not balanced.")


# Test case
# Input: "<>"
# Output: The brackets are balanced.

# Input: "<>"
# Output: The brackets are balanced.

# Input: "<><"
# Output: The brackets are balanced.

# Input: "<>"
# Output: The brackets are balanced.

# Input: "<"
# Output: The brackets are not balanced.

# Input: "<<"
# Output: The brackets are not balanced.

# Input: ">>"
# Output: The brackets are not balanced.