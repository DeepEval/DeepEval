import torch

class BayesianTabMlp(BaseBayesianModel):
    def __init__(
        self,
        column_idx: Dict[str, int],
        *,
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
        pred_dim: int = 1,
    ):
        super(BayesianTabMlp, self).__init__()

        # Initialize the layers and attributes as shown above

    def _get_embeddings(self, X: Tensor) -> Tensor:
        # Extract categorical embeddings
        if cat_embed_input is not None:
            cat_embeddings = self.cat_embed(X[:, cat_embed_input])

        # Normalize continuous features
        if continuous_cols is not None:
            normalized_continuous_features = self.cont_norm(X[:, continuous_cols])

        # Optionally embed normalized continuous features
        if cont_embed_dim and cont_embed_dropout:
            continuous_embeddings = self.cont_embed(
                normalized_continuous_features,
                dim=cont_embed_dim,
                dropout=cont_embed_dropout,
                activation=cont_embed_activation,
            )

        # Concatenate all tensors along the second dimension
        embeddings = torch.cat(
            (
                cat_embeddings,
                continuous_embeddings,
            ),
            dim=1,
        )

        return embeddings

if __name__ == "__main__":
    # Create sample input values
    X = torch.randn(10, 4)
    column_idx = {"column1": 0, "column2": 1, "column3": 2, "column4": 3}
    cat_embed_input = [("column1", 2, 4), ("column2", 2, 4)]
    continuous_cols = ["column3", "column4"]
    cont_embed_dim = 10
    cont_embed_dropout = 0.2
    cont_embed_activation = "leaky_relu"
    mlp_hidden_dims = [200, 100]
    mlp_activation = "leaky_relu"
    prior_sigma_1 = 1
    prior_sigma_2 = 0.002
    prior_pi = 0.8
    posterior_mu_init = 0.0
    posterior_rho_init = -7.0
    pred_dim = 1

    # Call the function and print the results
    model = BayesianTabMlp(column_idx, cat_embed_input, continuous_cols, cont_embed_dim,cont_embed_dropout, cont_embed_activation, mlp_hidden_dims, mlp_activation, prior_sigma_1,prior_sigma_2, prior_pi, posterior_mu_init, posterior_rho_init, pred_dim)
    embeddings = model._get_embeddings(X)
    print(embeddings.shape)