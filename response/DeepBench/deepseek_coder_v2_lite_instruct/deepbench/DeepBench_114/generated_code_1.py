import torch
import torch.nn as nn
import torch.optim as optim

class KimCNN(nn.Module):
    def __init__(self, vocabulary_size, max_seq_length, num_classes=2, out_channels=100, embed_dim=300, padding_idx=0, kernel_heights=[3, 4, 5], dropout=0.5, embedding_matrix=None, freeze_embedding_layer=False):
        super(KimCNN, self).__init__()
        
        # Initialize the embedding layer
        if embedding_matrix is None:
            self.embedding = nn.Embedding(vocabulary_size, embed_dim, padding_idx=padding_idx)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix), freeze=freeze_embedding_layer, padding_idx=padding_idx)
        
        # Initialize convolutional layers with varying kernel heights
        self.convs = nn.ModuleList([
            nn.Conv2d(1, out_channels, (kh, embed_dim), padding=(kh-1, 0)) for kh in kernel_heights
        ])
        
        # Initialize max-pooling layers corresponding to each convolutional layer
        self.pools = nn.ModuleList([
            nn.MaxPool1d(max_seq_length - kh + 1) for kh in kernel_heights
        ])
        
        # Initialize a dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Initialize a fully connected layer for classification
        self.fc = nn.Linear(len(kernel_heights) * out_channels, num_classes)

    def forward(self, x):
        # Embed the input
        x = self.embedding(x).unsqueeze(1)  # Add channel dimension
        
        # Apply convolutional layers
        conv_outputs = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        
        # Apply max-pooling layers
        pooled_outputs = [pool(conv_out).squeeze(2) for conv_out, pool in zip(conv_outputs, self.pools)]
        
        # Concatenate all pooled outputs
        h_pool = torch.cat(pooled_outputs, 1)
        
        # Apply dropout
        h_drop = self.dropout(h_pool)
        
        # Fully connected layer
        logits = self.fc(h_drop)
        
        return logits

if __name__ == "__main__":
    vocabulary_size = 5000
    max_seq_length = 100
    num_classes = 2
    embed_dim = 100
    kernel_heights = [3, 4, 5]
    dropout = 0.5
    batch_size = 32

    # Dummy input
    x = torch.randint(0, vocabulary_size, (batch_size, max_seq_length))

    # Create the model
    model = KimCNN(vocabulary_size, max_seq_length, num_classes, embed_dim, kernel_heights, dropout)

    # Print the model architecture
    print(model)

    # Forward pass
    output = model(x)
    print(output.shape)  # Should be (batch_size, num_classes)