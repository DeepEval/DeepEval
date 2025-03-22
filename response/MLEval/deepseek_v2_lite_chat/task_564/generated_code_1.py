import torch
import torch.nn as nn

def method():
    # Define the LSTM layer
    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(LSTM, self).__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size)
        
        def forward(self, input):
            lstm_out, _ = self.lstm(input)
            return lstm_out

    # Example usage
    input_size = 10  # Example input size
    hidden_size = 20  # Example hidden state size
    
    # Create the LSTM layer
    lstm_layer = LSTM(input_size, hidden_size)
    
    # Generate some input data
    input_data = torch.randn(50, 10, input_size)  # 50 samples, each with 10 features of size input_size
    
    # Forward pass through the LSTM layer
    output = lstm_layer(input_data)
    
    # Optionally, you can reshape or process the output as needed
    output_reshaped = output.view(-1, hidden_size)  # Reshape for example purposes
    
    return output_reshaped  # Return the reshaped output

# Call the method for validation
print(method())