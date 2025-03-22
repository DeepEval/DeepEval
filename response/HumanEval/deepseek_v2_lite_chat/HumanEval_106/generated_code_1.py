from math import factorial

def method():
    def f(n):
        # Initialize the list with zeros
        output = [0] * n
        
        # Calculate the factorial for even indices and sum for odd indices
        for i in range(n):
            if i % 2 == 0:
                output[i] = factorial(i)
            else:
                output[i] = sum(range(1, i + 1))
        
        return output

    return f(10)  # You can change the input size as needed