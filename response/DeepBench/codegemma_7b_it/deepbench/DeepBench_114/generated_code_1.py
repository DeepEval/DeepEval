import torch
import torch.nn as nn

class KimCNN(nn.Module):
    def __init__(self, vocabulary_size, max_seq_length, num_classes=2, out_channels=100, embed_dim=300, padding_idx=0, kernel_heights=[3, 4, 5], dropout=0.5, embedding_matrix=None, freeze_embedding_layer=False):
        super(KimCNN, self).__init__()

        # Initialize embedding layer
        if embedding_matrix is None:
            self.embedding = nn.Embedding(vocabulary_size, embed_dim, padding_idx=padding_idx)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=freeze_embedding_layer)

        # Initialize convolutional layers with varying kernel heights
        self.convs = nn.ModuleList([
            nn.Conv2d(1, out_channels, (k, embed_dim)) for k in kernel_heights
        ])

        # Initialize max-pooling layers corresponding to each convolutional layer
        self.max_pools = nn.ModuleList([
            nn.MaxPool2d((max_seq_length - k + 1, 1)) for k in kernel_heights
        ])

        # Initialize dropout layer
        self.dropout = nn.Dropout(dropout)

        # Initialize fully connected layer for classification
        self.fc = nn.Linear(out_channels * len(kernel_heights), num_classes)

    def forward(self, x):
        # Embed the input sequence
        x = self.embedding(x)

        # Reshape the embedded sequence for convolutional operations
        x = x.unsqueeze(1)

        # Apply convolutional, max-pooling, and ReLU activation for each kernel height
        conv_outputs = []
        for conv, max_pool in zip(self.convs, self.max_pools):
            conv_outputs.append(max_pool(torch.relu(conv(x))))

        # Concatenate the outputs from all convolutional layers
        concat_outputs = torch.cat(conv_outputs, dim=1)

        # Flatten the concatenated outputs
        concat_outputs = concat_outputs.view(-1, self.out_channels * len(self.kernel_heights))

        # Apply dropout and fully connected layer
        x = self.dropout(concat_outputs)
        x = self.fc(x)

        return x

# Minimal runnable example
if __name__ == "__main__":
    # Sample input values
    vocabulary_size = 10000
    max_seq_length = 100
    num_classes = 2
    out_channels = 100
    embed_dim = 300
    padding_idx = 0
    kernel_heights = [3, 4, 5]
    dropout = 0.5

    # Create an instance of the model
    model = KimCNN(vocabulary_size, max_seq_length, num_classes, out_channels, embed_dim, padding_idx, kernel_heights, dropout)

    # Print the model architecture
    print(model)

    # Sample input data
    input_data = torch.randint(0, vocabulary_size, (10, max_seq_length))

    # Pass the input data through the model
    output = model(input_data)

    # Print the model output
    print(output)