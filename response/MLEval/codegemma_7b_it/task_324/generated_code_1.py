import time

def method():
    start_time = time.time()

    # Code to be measured

    end_time = time.time()
    execution_time = end_time - start_time
    print("Method execution time:", execution_time)

    output = "Method output"
    return output

method()