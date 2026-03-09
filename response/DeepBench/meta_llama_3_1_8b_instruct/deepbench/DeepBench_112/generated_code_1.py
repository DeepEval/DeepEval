import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Literal

class BayesianTabMlp(nn.Module):
    def __init__(self, 
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
                 pred_dim: int = 1
                 ):
        super(BayesianTabMlp, self).__init__()
        self.column_idx = column_idx
        self.cat_embed_input = cat_embed_input
        self.cat_embed_activation = cat_embed_activation
        self.continuous_cols = continuous_cols
        self.embed_continuous = embed_continuous
        self.cont_embed_dim = cont_embed_dim
        self.cont_embed_dropout = cont_embed_dropout
        self.cont_embed_activation = cont_embed_activation
        self.use_cont_bias = use_cont_bias
        self.cont_norm_layer = cont_norm_layer
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.pred_dim = pred_dim
        
        # Initialize the layers and attributes
        self.cat_embed = nn.ModuleList()
        self.cont_norm = nn.ModuleList()
        self.cont_embed = nn.ModuleList()
        if cat_embed_input is not None:
            for embed_dim, embed_input in cat_embed_input:
                self.cat_embed.append(nn.Embedding(embed_input, embed_dim))
        if continuous_cols is not None:
            for _ in continuous_cols:
                self.cont_norm.append(nn.LayerNorm(1))
                self.cont_embed.append(nn.Linear(1, self.cont_embed_dim))
                if self.cont_embed_dropout is not None:
                    self.cont_embed.append(nn.Dropout(self.cont_embed_dropout))
        
    def _get_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        embeddings = []
        if self.cat_embed_input is not None:
            for cat_embed in self.cat_embed:
                embeddings.append(cat_embed(X[:, self.column_idx['cat']]))
            embeddings = torch.cat(embeddings, dim=1)
        if self.continuous_cols is not None:
            continuous_features = X[:, self.column_idx['cont']]
            for cont_norm, cont_embed in zip(self.cont_norm, self.cont_embed):
                continuous_features = cont_norm(continuous_features)
                continuous_features = cont_embed(continuous_features)
                embeddings = torch.cat((embeddings, continuous_features), dim=1)
        return embeddings

def function_name(args): 
    # Your code here
    X = torch.randn(10, 5)
    model = BayesianTabMlp(column_idx={'cat': [0, 1, 2], 'cont': [3, 4]})
    return model._get_embeddings(X)

if __name__ == "__main__":
    print(function_name(None))