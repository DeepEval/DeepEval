import torch
import torch.nn as nn
import torch.nn.functional as F

class KimCNN(nn.Module):
    def __init__(self, vocabulary_size, max_seq_length, num_classes=2, out_channels=100, embed_dim=300, padding_idx=0, kernel_heights=[3, 4, 5], dropout=0.5, embedding_matrix=None, freeze_embedding_layer=False):
        super(KimCNN, self).__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=freeze_embedding_layer)
        else:
            self.embedding = nn.Embedding(vocabulary_size, embed_dim, padding_idx=padding_idx)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, out_channels, (kh, embed_dim)) for kh in kernel_heights
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_heights) * out_channels, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        conv_results = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled_results = [F.max_pool1d(cr, cr.size(2)).squeeze(2) for cr in conv_results]
        x = torch.cat(pooled_results, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    vocabulary_size = 5000
    max_seq_length = 100
    num_classes = 2
    out_channels = 100
    embed_dim = 300
    kernel_heights = [3, 4, 5]
    dropout = 0.5
    embedding_matrix = None  # or provide a suitable embedding matrix if needed
    freeze_embedding_layer = False

    model = KimCNN(vocabulary_size, max_seq_length, num_classes, out_channels, embed_dim, kernel_heights=kernel_heights, dropout=dropout, embedding_matrix=embedding_matrix, freeze_embedding_layer=freeze_embedding_layer)

    sample_input = torch.randint(0, vocabulary_size, (32, max_seq_length))  # Batch size of 32
    output = model(sample_input)
    print(output.shape)  # Expected output shape: (32, num_classes)