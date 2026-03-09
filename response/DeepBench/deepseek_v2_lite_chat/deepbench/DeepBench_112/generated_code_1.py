import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler

class BayesianTabMlp(torch.nn.Module):
    def __init__(self, column_idx: Dict[str, int], *, cat_embed_input: Optional[List[Tuple[str, int, int]]] = None, cat_embed_activation: Optional[str] = None, continuous_cols: Optional[List[str]] = None, embed_continuous: Optional[bool] = None, cont_embed_dim: Optional[int] = None, cont_embed_dropout: Optional[float] = None, cont_embed_activation: Optional[str] = None, use_cont_bias: Optional[bool] = None, cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]] = None, mlp_hidden_dims: List[int] = [200, 100], mlp_activation: str = "leaky_relu", prior_sigma_1: float = 1, prior_sigma_2: float = 0.002, prior_pi: float = 0.8, posterior_mu_init: float = 0.0, posterior_rho_init: float = -7.0, pred_dim=1):
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

        self.cat_embed = torch.nn.ModuleList()
        if self.cat_embed_input is not None:
            for cat, start, end in self.cat_embed_input:
                embedding = torch.nn.Embedding(end - start + 1, self.cont_embed_dim)
                self.cat_embed.append(embedding)

        self.cont_embed = torch.nn.ModuleList()
        if self.continuous_cols is not None:
            scaler = StandardScaler()
            X_cont = X[:, self.continuous_cols].astype(float)
            X_cont_scaled = scaler.fit_transform(X_cont)
            X_cont_embed = torch.nn.ModuleList()
            if self.cont_embed_dim is not None:
                for i in range(len(self.continuous_cols)):
                    embedding = torch.nn.Embedding(len(X_cont_scaled[i]), self.cont_embed_dim)
                    X_cont_embed.append(embedding)
            else:
                for i in range(len(self.continuous_cols)):
                    embedding = torch.nn.Embedding(len(X_cont_scaled[i]), 1)
                    X_cont_embed.append(embedding)
            self.cont_embed = X_cont_embed

    def _get_embeddings(self, X: Tensor) -> Tensor:
        X_cat = X[:, self.column_idx.get(torch.nn.Category.__name__, None)]
        if self.cat_embed_input is not None:
            for embedding in self.cat_embed:
                X_cat = torch.cat((X_cat, embedding(X_cat)), dim=1)
        if self.continuous_cols is not None:
            X_cont = X[:, self.column_idx.get(torch.nn.Continuous.__name__, None)]
            if self.cont_embed_dim is not None:
                for embedding in self.cont_embed:
                    X_cont = torch.cat((X_cont, embedding(X_cont.long())), dim=1)
            else:
                for embedding in self.cont_embed:
                    X_cont = torch.cat((X_cont, embedding(X_cont.long())), dim=1)
            X_cat = torch.cat((X_cat, X_cont), dim=1)
        return X_cat
    
if __name__ == "__main__":
    column_idx = {"column1": 0, "column2": 1, "column3": 2, "column4": 3}
    cat_embed_input = [("column1", 0, 2), ("column2", 0, 2)]
    continuous_cols = ["column3", "column4"]
    model = BayesianTabMlp(column_idx, cat_embed_input=cat_embed_input, continuous_cols=continuous_cols, cont_embed_dim=4)
    X = torch.randn(10, 4)
    embeddings = model._get_embeddings(X)
    print(embeddings.shape)