import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Literal

class BaseBayesianModel(nn.Module):
    def __init__(self):
        super(BaseBayesianModel, self).__init__()

class BayesianTabMlp(BaseBayesianModel):
    def __init__(self, column_idx: Dict[str, int], *, cat_embed_input: Optional[List[Tuple[str, int, int]]] = None, 
                 cat_embed_activation: Optional[str] = None, continuous_cols: Optional[List[str]] = None, 
                 embed_continuous: Optional[bool] = None, cont_embed_dim: Optional[int] = None, 
                 cont_embed_dropout: Optional[float] = None, cont_embed_activation: Optional[str] = None, 
                 use_cont_bias: Optional[bool] = None, cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]] = None, 
                 mlp_hidden_dims: List[int] = [200, 100], mlp_activation: str = "leaky_relu", 
                 prior_sigma_1: float = 1, prior_sigma_2: float = 0.002, prior_pi: float = 0.8, 
                 posterior_mu_init: float = 0.0, posterior_rho_init: float = -7.0, pred_dim=1):
        super(BayesianTabMlp, self).__init__()
        self.cat_embed_input = cat_embed_input
        self.continuous_cols = continuous_cols
        
        if cat_embed_input is not None:
            self.cat_embed = nn.ModuleList(
                [nn.Embedding(num_embeddings=cat[1], embedding_dim=cat[2]) for cat in cat_embed_input]
            )
        
        if continuous_cols is not None and embed_continuous:
            self.cont_norm = nn.BatchNorm1d(len(continuous_cols)) if cont_norm_layer == "batchnorm" else nn.LayerNorm(len(continuous_cols))
            self.cont_embed = nn.Linear(len(continuous_cols), cont_embed_dim)
        
    def _get_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        embeddings = []
        
        if self.cat_embed_input is not None:
            cat_embeddings = [self.cat_embed[i](X[:, self.column_idx[cat[0]]]) for i, cat in enumerate(self.cat_embed_input)]
            embeddings.append(torch.cat(cat_embeddings, dim=1))
        
        if self.continuous_cols is not None:
            continuous_features = X[:, [self.column_idx[col] for col in self.continuous_cols]]
            normalized_cont = self.cont_norm(continuous_features)
            if hasattr(self, 'cont_embed'):
                cont_embeddings = self.cont_embed(normalized_cont)
                embeddings.append(cont_embeddings)
            else:
                embeddings.append(normalized_cont)
        
        return torch.cat(embeddings, dim=1)

if __name__ == "__main__":
    column_idx = {'cat1': 0, 'cat2': 1, 'cont1': 2, 'cont2': 3}
    cat_embed_input = [('cat1', 5, 3), ('cat2', 10, 4)]
    continuous_cols = ['cont1', 'cont2']
    
    model = BayesianTabMlp(column_idx, cat_embed_input=cat_embed_input, continuous_cols=continuous_cols, 
                           embed_continuous=True, cont_embed_dim=2, cont_norm_layer="batchnorm")
    
    # Create a sample input tensor (batch_size=2, num_features=4)
    X = torch.tensor([[1, 2, 0.5, 1.5], [0, 1, 0.2, 0.8]], dtype=torch.float32)
    
    # Get embeddings
    embeddings = model._get_embeddings(X)
    print(embeddings)