import torch
import torch.nn as nn

class KimCNN(nn.Module):
    def __init__(self, vocabulary_size, max_seq_length, num_classes=2, out_channels=100, embed_dim=300, padding_idx=0, kernel_heights=[3, 4, 5], dropout=0.5, embedding_matrix=None, freeze_embedding_layer=False):
        super(KimCNN, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix if embedding_matrix is not None else torch.zeros(vocabulary_size, embed_dim)))

        self.conv = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=k) for k in kernel_heights])
        
        max_pool_sizes = [x.data.cpu().item() for x in torch.ops.torch.nn.modules.max_pool.max_pool1d.max_pool_forward(torch.tensor(list(range(1, len(kernel_heights) + 1))).unsqueeze(1), kernel_sizes=torch.tensor(kernel_heights))]
        self.pool = nn.ModuleList([nn.MaxPool1d(x) for x in max_pool_sizes])

        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Linear(in_features=len(kernel_heights) * out_channels, out_features=num_classes)

        self.freeze_embedding_layer = freeze_embedding_layer

        if self.freeze_embedding_layer:
            for param in self.embedding.parameters():
                param.requires_grad = False

    def forward(self, text):
        embedded = self.embedding(text)

        # Convolutional layers
        embedded = [F.relu(conv(embedded.squeeze(3))).squeeze(3) for conv in self.conv]
        embedded = torch.cat(embedded, 1)

        # Max-pooling
        embedded = torch.unsqueeze(embedded, 1)
        embedded = [F.relu(pool(embedded)).squeeze(3) for pool in self.pool]

        embedded = torch.cat(embedded, 1)

        # Dropout
        embedded = self.dropout(embedded)

        # Classification
        output = self.fc(embedded)

        return output


def run_example():
    vocabulary_size = 1000
    max_seq_length = 100
    num_classes = 2
    out_channels = 256
    embed_dim = 100
    padding_idx = 0
    kernel_heights = [3, 4, 5]
    dropout = 0.2
    embedding_matrix = torch.rand(vocabulary_size, embed_dim)
    freeze_embedding_layer = False

    model = KimCNN(vocabulary_size, max_seq_length, num_classes, out_channels, embed_dim, padding_idx, kernel_heights, dropout, embedding_matrix, freeze_embedding_layer)

    text = torch.randint(0, vocabulary_size, (1, max_seq_length))

    output = model(text)

    print(output.shape)


if __name__ == "__main__":
    run_example()