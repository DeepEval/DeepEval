def method():
    import numpy as np

    def fibfib(n):
        if n == 0:
            return 0
        elif n == 1:
            return 0
        elif n == 2:
            return 1
        else:
            sequence = np.zeros(n+1)
            sequence[0] = 0
            sequence[1] = 0
            sequence[2] = 1
            
            for i in range(3, n+1):
                sequence[i] = sequence[i-1] + sequence[i-2] + sequence[i-3]
                
            return sequence[n]
    
    return fibfib(n)