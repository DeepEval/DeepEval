import torch
import torch.nn as nn

class KimCNN(nn.Module):
    def __init__(self, vocabulary_size, max_seq_length, num_classes=2, out_channels=100,
  embed_dim=300, padding_idx=0, kernel_heights=[3, 4, 5], dropout=0.5, embedding_matrix=None,
  freeze_embedding_layer=False): 
        super(KimCNN, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embed_dim, padding_idx=padding_idx)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        if freeze_embedding_layer:
            self.embedding.weight.requires_grad = False
        self.convs = nn.ModuleList([nn.Conv2d(1, out_channels, (k, embed_dim))
  for k in kernel_heights])
        self.maxpools = nn.ModuleList([nn.AdaptiveMaxPool2d((1, 1)) for _ in kernel_heights])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_heights) * out_channels, num_classes)
    

if __name__ == "__main__":
    vocab_size = 10000
    max_seq_length = 100
    model = KimCNN(vocab_size, max_seq_length)
    input_tensor = torch.randint(0, vocab_size, (2, max_seq_length))
    output = model(input_tensor)
    print(output.shape) 