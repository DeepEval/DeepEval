import numpy as np

def _coefficients_no_filter(f, degree, use_broadcasting=True):
  d = np.array(degree)
  n = np.arange(-d, d + 1)
  if use_broadcasting:
    return np.exp(-1j * 2 * np.pi * n[:, None, None] * np.arange(2 * np.pi)[:, None, None] / (2 * np.pi)) @ f(np.arange(2 * np.pi)[:, None, None]) 
  else:
    coefficients = np.zeros(2 * d + 1, dtype=complex)
    for i in range(2 * d + 1):
      coefficients[i] = (1/(2 * np.pi)) * np.trapz(f(np.arange(2 * np.pi)) * np.exp(-1j * 2 * np.pi * n[i] * np.arange(2 * np.pi) / (2 * np.pi)), np.arange(2 * np.pi))
    return coefficients

if __name__ == "__main__":
  f = lambda x: np.sin(x)
  degree = 5
  coeffs = _coefficients_no_filter(f, degree)
  print(coeffs)