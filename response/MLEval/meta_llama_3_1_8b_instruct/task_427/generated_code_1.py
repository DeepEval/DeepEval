import cmdstanpy

# Load the STAN model
# model = cmdstanpy.CmdStanModel(stan_file='path_to_your_model.stan')

# 修改为本地数据文件
model = cmdstanpy.CmdStanModel(stan_file='evaluation/dynamic_checking/baselines/MLEval/meta_llama_3_1_8b_instruct/testcases/task_427.stan')

# Define the method function
def method():
    # Define the data dictionary
    data = {
        'N': 100,  # number of observations
        'y': [1, 2, 3, 4, 5],  # observations
        'a': 1,  # parameter a
        'b': 2  # parameter b
    }

    # Fit the model
    fit = model.vb(data=data)

    # Print the output
    print(fit.summary())

    # Return the output
    return fit.summary()

# Call the method function for validation
output = method()

# Print the output if needed
print(output)