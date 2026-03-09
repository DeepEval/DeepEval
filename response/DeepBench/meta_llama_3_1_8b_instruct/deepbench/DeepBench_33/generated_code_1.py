import torch

def _create_identity_grid(args):
    height, width = args
    grid = torch.linspace(0, 1, steps=width, device=torch.device('cpu'))  # Create x-coordinates
    grid = grid.reshape(1, 1, -1).expand(1, height, -1)  # Repeat for each row
    grid_y, grid_x = torch.meshgrid(grid, torch.linspace(0, 1, steps=height, device=torch.device('cpu')))  # Create y-coordinates
    return torch.cat((grid_x, grid_y), dim=1).float()  # Stack x and y coordinates

if __name__ == "__main__":
    # Create sample input values
    height, width = 10, 10
    input_args = [height, width]
    
    # Call the function
    output = _create_identity_grid(input_args)
    
    # Print the results
    print(output)