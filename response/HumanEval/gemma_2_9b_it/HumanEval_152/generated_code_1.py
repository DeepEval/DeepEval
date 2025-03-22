def method(scores, guesses):
  output = []
  for i in range(len(scores)):
    difference = abs(scores[i] - guesses[i])
    output.append(difference)
  return output

# Test case
scores1 = [1, 2, 3, 4, 5, 1]
guesses1 = [1, 2, 3, 4, 2, -2]
output1 = method(scores1, guesses1)
print(f"Test Case 1: {output1}")  # Expected output: [0, 0, 0, 0, 3, 3]

scores2 = [0, 5, 0, 0, 0, 4]
guesses2 = [4, 1, 1, 0, 0, -2]
output2 = method(scores2, guesses2)
print(f"Test Case 2: {output2}")  # Expected output: [4, 4, 1, 0, 0, 6]