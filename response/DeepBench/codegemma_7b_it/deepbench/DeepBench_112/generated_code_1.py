from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianTabMlp(nn.Module):
    # ...

    def _get_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        embeddings = []

        # Extract categorical embeddings
        if self.cat_embed_input is not None:
            for name, in_size, out_size in self.cat_embed_input:
                embeddings.append(self.cat_embed[name](X[:, self.column_idx[name]].long()))

        # Normalize and embed continuous features
        if self.continuous_cols is not None:
            x_cat = X[:, self.column_idx["cat"]]
            x_cont = X[:, self.column_idx["cont"]]

            x_cont_norm = self.cont_norm(x_cont)
            if self.embed_continuous:
                x_cont_embed = self.cont_embed(x_cont_norm)
                embeddings.append(x_cont_embed)
            else:
                embeddings.append(x_cont_norm)

        # Concatenate embeddings
        embeddings = torch.cat(embeddings, dim=1)

        return embeddings

# Minimal runnable example
if __name__ == "__main__":
    # Sample input data
    column_idx = {"cat": 0, "cont": 1}
    X = torch.tensor([[1, 2.5], [0, 0.5], [2, -1.5]], dtype=torch.float32)

    # Create an instance of BayesianTabMlp
    model = BayesianTabMlp(column_idx)

    # Get embeddings
    embeddings = model._get_embeddings(X)

    # Print embeddings
    print(embeddings)