Here is a minimal runnable example that creates sample input values, calls the function, and prints the results:

if __name__ == "__main__":
    # Create sample input values
    input_values = torch.rand(5, 1)

    # Call the function
    model = FCNN(n_input_units=1, n_output_units=1, hidden_units=(32, 32))
    output = model(input_values)

    # Print the results
    print(output)