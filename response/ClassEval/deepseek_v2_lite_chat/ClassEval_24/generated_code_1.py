class ComplexCalculator:
    def __init__(self, real=0, imag=0):
        self.real = real
        self.imag = imag

    @staticmethod
    def add(c1, c2):
        real = c1.real + c2.real
        imag = c1.imag + c2.imag
        return ComplexCalculator(real, imag)

    @staticmethod
    def subtract(c1, c2):
        real = c1.real - c2.real
        imag = c1.imag - c2.imag
        return ComplexCalculator(real, imag)

    @staticmethod
    def multiply(c1, c2):
        real = c1.real * c2.real - c1.imag * c2.imag
        imag = c1.real * c2.imag + c1.imag * c2.real
        return ComplexCalculator(real, imag)

    @staticmethod
    def divide(c1, c2):
        # Implementing complex division
        denominator_real = c2.real * c2.real + c2.imag * c2.imag
        denominator_imag = c2.real * c2.imag - c1.real * c1.imag
        real = (c1.real * c2.real + c1.imag * c2.imag) / denominator_real
        imag = (c1.imag * c2.real - c1.real * c2.imag) / denominator_real
        return ComplexCalculator(real, imag)

# Test cases
if __name__ == "__main__":
    c1 = ComplexCalculator(1, 2)
    c2 = ComplexCalculator(3, 4)

    print("Addition:", c1.add(c1, c2))  # (4+6j)
    print("Subtraction:", c1.subtract(c1, c2))  # (-2-2j)
    print("Multiplication:", c1.multiply(c1, c2))  # (-5+10j)
    print("Division:", c1.divide(c1, c2))  # (0.44+0.08j)