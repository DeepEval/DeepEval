import re

def method():
  # text = input("Enter text: ")
  
  # 修改
  text = "I am bored. I am very bored. I am so bored I can hardly stand it."

  boredom_count = 0

  # Split the text into sentences using regex
  sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text)

  # Iterate through each sentence
  for sentence in sentences:
    if sentence.startswith("I"):
      boredom_count += 1

  return boredom_count


# Test case
test_text = "I am bored. I am very bored. I am so bored I can hardly stand it."
print(method())