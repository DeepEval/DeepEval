import math

def frequencies_to_period(frequencies, decimals=0):
  frequencies = [round(freq, decimals) for freq in frequencies]
  gcd = frequencies[0]
  for i in range(1, len(frequencies)):
    gcd = math.gcd(gcd, frequencies[i])
  return (2 * math.pi) / gcd 

if __name__ == "__main__":
  frequencies = (2, 4, 6)
  period = frequencies_to_period(frequencies)
  print(f"The period is: {period}")