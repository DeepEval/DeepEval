import torch
import torch.nn as nn

class KimCNN(nn.Module):
    def __init__(self, vocabulary_size, max_seq_length, num_classes=2, out_channels=100, embed_dim=300, padding_idx=0, kernel_heights=[3, 4, 5], dropout=0.5, embedding_matrix=None, freeze_embedding_layer=False):
        super(KimCNN, self).__init__()

        # Initialize the embedding layer
        self.embedding_layer = nn.Embedding(vocabulary_size, embed_dim, padding_idx=padding_idx)
        if embedding_matrix is not None:
            self.embedding_layer.weight.data.copy_(embedding_matrix)
            self.embedding_layer.weight.requires_grad = not freeze_embedding_layer

        # Initialize convolutional layers with varying kernel heights
        self.convolutional_layers = nn.ModuleList([
            nn.Conv1d(embed_dim, out_channels, kernel_size=kernel_heights[i], stride=1, padding=kernel_heights[i] // 2)
            for i in range(len(kernel_heights))
        ])

        # Initialize max-pooling layers corresponding to each convolutional layer
        self.max_pooling_layers = nn.ModuleList([
            nn.MaxPool1d(kernel_size=kernel_heights[i], stride=1, padding=kernel_heights[i] // 2)
            for i in range(len(kernel_heights))
        ])

        # Initialize a dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Initialize a fully connected layer for classification
        self.fully_connected_layer = nn.Linear(len(kernel_heights) * out_channels, num_classes)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = torch.relu(x)
        for convolutional_layer, max_pooling_layer in zip(self.convolutional_layers, self.max_pooling_layers):
            x = convolutional_layer(x)
            x = max_pooling_layer(x)
        x = x.view(-1, self.fully_connected_layer.in_features)
        x = self.dropout_layer(x)
        x = self.fully_connected_layer(x)
        return x

if __name__ == "__main__":
    # Create sample input values
    input_data = torch.randint(0, 100, (10, 5))

    # Create the model and forward pass
    model = KimCNN(vocabulary_size=100, max_seq_length=5, num_classes=2, out_channels=100,embed_dim=300, padding_idx=0, kernel_heights=[3, 4, 5], dropout=0.5, embedding_matrix=None,freeze_embedding_layer=False)
    output = model(input_data)

    # Print the output
    print(output)