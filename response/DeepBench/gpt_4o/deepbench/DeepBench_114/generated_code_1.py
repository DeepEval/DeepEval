import torch
import torch.nn as nn
import torch.nn.functional as F

class KimCNN(nn.Module):
    def __init__(self, vocabulary_size, max_seq_length, num_classes=2, out_channels=100, embed_dim=300, padding_idx=0, kernel_heights=[3, 4, 5], dropout=0.5, embedding_matrix=None, freeze_embedding_layer=False):
        super(KimCNN, self).__init__()
        
        # Initialize the embedding layer
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=freeze_embedding_layer, padding_idx=padding_idx)
        else:
            self.embedding = nn.Embedding(vocabulary_size, embed_dim, padding_idx=padding_idx)
        
        # Initialize convolutional layers with varying kernel heights
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, out_channels, (k, embed_dim)) for k in kernel_heights
        ])
        
        # Initialize dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Initialize a fully connected layer for classification
        self.fc = nn.Linear(len(kernel_heights) * out_channels, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # (batch_size, 1, max_seq_length, embed_dim)
        
        # Apply each convolution and max-pooling
        conv_results = [F.relu(conv(x)).squeeze(3) for conv in self.conv_layers]  # [(batch_size, out_channels, W), ...] * len(kernel_heights)
        pooled_results = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in conv_results]  # [(batch_size, out_channels), ...]
        
        # Concatenate the pooled features
        cat = torch.cat(pooled_results, 1)  # (batch_size, len(kernel_heights) * out_channels)
        
        # Apply dropout
        out = self.dropout(cat)
        
        # Pass through fully connected layer
        out = self.fc(out)  # (batch_size, num_classes)
        
        return out

if __name__ == "__main__":
    # Define a simple example to test the class
    vocabulary_size = 5000
    max_seq_length = 50
    num_classes = 2
    embed_dim = 300
    batch_size = 4
    example_input = torch.randint(0, vocabulary_size, (batch_size, max_seq_length))

    # Initialize the model
    model = KimCNN(vocabulary_size=vocabulary_size, max_seq_length=max_seq_length, num_classes=num_classes, embed_dim=embed_dim)

    # Forward pass with the example input
    output = model(example_input)

    # Print the output
    print("Output shape:", output.shape)
    print("Output:", output)