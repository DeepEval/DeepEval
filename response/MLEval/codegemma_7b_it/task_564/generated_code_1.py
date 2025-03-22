import torch
import torch.nn as nn

# Define the LSTM cell
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()

        # Define the weights for input, hidden and cell states
        self.weight_ih = nn.Parameter(torch.randn(input_size + hidden_size, hidden_size))
        self.weight_hh = nn.Parameter(torch.randn(hidden_size + hidden_size, hidden_size))

        # Define the biases for input, hidden and cell states
        self.bias_ih = nn.Parameter(torch.randn(hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(hidden_size))

    def forward(self, input_, hidden):
        # Concatenate the input and previous hidden state
        combined = torch.cat((input_, hidden), dim=1)

        # Calculate the new cell state
        cell = torch.sigmoid(torch.mm(combined, self.weight_ih) + self.bias_ih)

        # Calculate the new hidden state
        hidden = torch.sigmoid(torch.mm(torch.cat((cell, hidden), dim=1), self.weight_hh) + self.bias_hh)

        return hidden

def method():

    # Define the input size and hidden state size
    input_size = 10
    hidden_size = 20

    # Create an instance of the LSTM cell
    lstm_cell = LSTMCell(input_size, hidden_size)

    # Define the input and hidden state
    input_ = torch.randn(1, input_size)
    hidden = torch.randn(1, hidden_size)

    # Run the LSTM cell forward
    output = lstm_cell(input_, hidden)

    return output

# Call the method and print the output
output = method()
print(output)