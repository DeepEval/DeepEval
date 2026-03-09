import torch

def xavier_normal_(tensor, gain=1.0, generator=None):
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    with torch.random.fork_rng():
        if generator is not None:
            torch.random.set_generator_states(generator, set_current=True)
        return torch.nn.init.normal_(tensor, 0.0, std)

if __name__ == "__main__":
    # Create a sample tensor
    tensor = torch.randn(3, 3)
    
    # Call the function
    xavier_normal_(tensor)
    
    # Print the results
    print(tensor)