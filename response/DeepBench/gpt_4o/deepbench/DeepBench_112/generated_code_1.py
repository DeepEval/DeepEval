import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from typing_extensions import Literal

class BaseBayesianModel(nn.Module):
    def __init__(self):
        super(BaseBayesianModel, self).__init__()

class BayesianTabMlp(BaseBayesianModel):
    def __init__(
        self, 
        column_idx: Dict[str, int], 
        cat_embed_input: Optional[List[Tuple[str, int, int]]] = None, 
        cat_embed_activation: Optional[str] = None, 
        continuous_cols: Optional[List[str]] = None, 
        embed_continuous: Optional[bool] = None, 
        cont_embed_dim: Optional[int] = None, 
        cont_embed_dropout: Optional[float] = None, 
        cont_embed_activation: Optional[str] = None, 
        use_cont_bias: Optional[bool] = None, 
        cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]] = None, 
        mlp_hidden_dims: List[int] = [200, 100], 
        mlp_activation: str = "leaky_relu", 
        prior_sigma_1: float = 1, 
        prior_sigma_2: float = 0.002, 
        prior_pi: float = 0.8, 
        posterior_mu_init: float = 0.0, 
        posterior_rho_init: float = -7.0, 
        pred_dim=1, 
    ):
        super(BayesianTabMlp, self).__init__()
        
        self.column_idx = column_idx
        self.cat_embed_input = cat_embed_input
        self.continuous_cols = continuous_cols
        self.embed_continuous = embed_continuous

        # Initialize embeddings for categorical features
        if self.cat_embed_input:
            self.cat_embed = nn.ModuleDict({
                name: nn.Embedding(num_cat, embed_dim) 
                for name, num_cat, embed_dim in self.cat_embed_input
            })
        
        # Initialize normalization and embedding for continuous features
        if self.continuous_cols:
            self.cont_norm = nn.ModuleDict({
                col: nn.BatchNorm1d(1) if cont_norm_layer == "batchnorm" else nn.LayerNorm(1)
                for col in self.continuous_cols
            })
            if self.embed_continuous:
                self.cont_embed = nn.ModuleDict({
                    col: nn.Linear(1, cont_embed_dim) for col in self.continuous_cols
                })

    def _get_embeddings(self, X: Tensor) -> Tensor:
        embeddings = []
        
        if self.cat_embed_input:
            for name, _, _ in self.cat_embed_input:
                col_idx = self.column_idx[name]
                embeddings.append(self.cat_embed[name](X[:, col_idx].long()))
        
        if self.continuous_cols:
            for col in self.continuous_cols:
                col_idx = self.column_idx[col]
                cont_data = X[:, col_idx].unsqueeze(1).float()
                normalized = self.cont_norm[col](cont_data)
                if self.embed_continuous:
                    normalized = self.cont_embed[col](normalized)
                embeddings.append(normalized)
        
        return torch.cat(embeddings, dim=1)

if __name__ == "__main__":
    column_idx = {"cat1": 0, "cat2": 1, "cont1": 2, "cont2": 3}
    cat_embed_input = [("cat1", 5, 3), ("cat2", 10, 4)]
    continuous_cols = ["cont1", "cont2"]
    model = BayesianTabMlp(column_idx=column_idx, 
                           cat_embed_input=cat_embed_input, 
                           continuous_cols=continuous_cols, 
                           embed_continuous=True, 
                           cont_embed_dim=5, 
                           cont_norm_layer="batchnorm")
    
    X = torch.tensor([[1, 3, 2.0, 4.0], [2, 5, 1.0, 3.0]])
    embeddings = model._get_embeddings(X)
    print(embeddings)