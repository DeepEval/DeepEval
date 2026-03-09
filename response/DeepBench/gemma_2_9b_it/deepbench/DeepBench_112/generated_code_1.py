import torch
from torch import Tensor

class BayesianTabMlp(BaseBayesianModel):
    def __init__( self, column_idx: Dict[str, int], *, cat_embed_input: Optional[List[Tuple[str, int, int]]] = None, cat_embed_activation: Optional[str] = None, continuous_cols: Optional[List[str]] = None, embed_continuous: Optional[bool] = None, cont_embed_dim: Optional[int] = None, cont_embed_dropout: Optional[float] = None, cont_embed_activation: Optional[str] = None, use_cont_bias: Optional[bool] = None, cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]] = None, mlp_hidden_dims: List[int] = [200, 100], mlp_activation: str = "leaky_relu", prior_sigma_1: float = 1, prior_sigma_2: float = 0.002, prior_pi: float = 0.8, posterior_mu_init: float = 0.0, posterior_rho_init: float = -7.0, pred_dim=1, ):
        super(BayesianTabMlp, self).__init__()
        # Initialize the layers and attributes as shown above
    def _get_embeddings(self, X: Tensor) -> Tensor:
        embeddings = []
        if cat_embed_input is not None:
            for col, embed_dim, _ in cat_embed_input:
                embeddings.append(self.cat_embed(X[:, column_idx[col]]))
        if continuous_cols is not None:
            cont_tensor = X[:, [column_idx[col] for col in continuous_cols]]
            if self.cont_norm is not None:
                cont_tensor = self.cont_norm(cont_tensor)
            if self.embed_continuous:
                embeddings.append(self.cont_embed(cont_tensor))
        return torch.cat(embeddings, dim=1)


if __name__ == "__main__":
    X = torch.randn(10, 5)
    column_idx = {'col1': 0, 'col2': 1, 'col3': 2, 'col4': 3, 'col5': 4}
    model = BayesianTabMlp(column_idx=column_idx)
    embeddings = model._get_embeddings(X)
    print(embeddings.shape)