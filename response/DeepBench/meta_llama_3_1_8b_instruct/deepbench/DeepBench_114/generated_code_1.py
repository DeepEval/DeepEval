import torch
import torch.nn as nn
import torch.nn.functional as F

class KimCNN(nn.Module):
    def __init__(self, vocabulary_size, max_seq_length, num_classes=2, out_channels=100, embed_dim=300, padding_idx=0, kernel_heights=[3, 4, 5], dropout=0.5, embedding_matrix=None, freeze_embedding_layer=False):
        super(KimCNN, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embed_dim, padding_idx=padding_idx)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            if freeze_embedding_layer:
                for param in self.embedding.parameters():
                    param.requires_grad = False
        self.convolutions = nn.ModuleList([nn.Conv2d(1, out_channels, (kernel_height, embed_dim)) for kernel_height in kernel_heights])
        self.max_pool = nn.ModuleList([nn.MaxPool2d(kernel_height) for kernel_height in kernel_heights])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_channels * len(kernel_heights), num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add channel dimension
        conv_features = []
        for conv, pool in zip(self.convolutions, self.max_pool):
            feature = F.relu(conv(x))
            feature = pool(feature)
            conv_features.append(feature.squeeze(2))
        x = torch.cat(conv_features, 1)
        x = self.dropout(x)
        x = x.view(-1, x.size(1) * x.size(2))
        x = self.fc(x)
        return x

if __name__ == "__main__":
    vocabulary_size = 10000
    max_seq_length = 200
    num_classes = 2
    out_channels = 100
    embed_dim = 300
    padding_idx = 0
    kernel_heights = [3, 4, 5]
    dropout = 0.5
    embedding_matrix = None  # You can replace this with your own embedding matrix
    freeze_embedding_layer = False

    model = KimCNN(vocabulary_size, max_seq_length, num_classes, out_channels, embed_dim, padding_idx, kernel_heights, dropout, embedding_matrix, freeze_embedding_layer)
    input = torch.randint(0, vocabulary_size, (1, max_seq_length))
    output = model(input)
    print(output.shape)