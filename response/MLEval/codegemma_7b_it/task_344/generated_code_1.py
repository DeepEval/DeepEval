import pysmc

def method():
    model = pysmc.Model()

    # Create a normal distribution for x
    x = model.add_normal('x', mu=0, sd=1)

    # Create a normal distribution for y
    y = model.add_normal('y', mu=x, sd=0.5)

    # Set the mean of x to 5
    model.set_start_value(x, 5)

    # Optimize the model
    model.optimize(maxiter=1000)

    # Print the optimized value of x
    print(model.get_value('x'))

    # Return the output (optional)
    return model.get_value('x')

# Call the method to validate
method()