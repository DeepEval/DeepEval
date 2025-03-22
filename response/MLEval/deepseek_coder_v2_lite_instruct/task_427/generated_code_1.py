import pystan

def method():
    # Define the STAN model
    stan_model_code = """
    data {
        int<lower=0> N;
        vector[N] x;
        vector[N] y;
    }
    parameters {
        real alpha;
        real beta;
        real<lower=0> sigma;
    }
    model {
        y ~ normal(alpha + beta * x, sigma);
    }
    """
    
    # Compile the model with the VB method for speed
    model = pystan.StanModel(model_code=stan_model_code, model_name="linear_regression", method="variational")
    
    # Example data
    data = {
        'N': 100,
        'x': [1, 2, 3, 4, 5],
        'y': [2, 3, 4, 5, 6]
    }
    
    # Run the model with VB
    fit = model.vb(data=data)
    
    # Extract the output
    output = fit.extract()
    
    return output

# Call the method for validation
output = method()
print(output)