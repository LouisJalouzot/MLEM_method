from torch import nn
import torch.nn.functional as F
from torchsort import soft_rank


class BaseModel(nn.Module):
    def __init__(
        self, loss="spearman", regularization="l2", regularization_strength=1
    ):
        super().__init__()
        self.regularization = regularization
        self.regularization_strength = regularization_strength
        if loss == "mse":
            self.loss = self.mse
        elif loss == "spearman":
            self.loss = self.spearman_diff

    def compute_gradient_norm(self, norm_type=2):
        """
        Computes the norm of gradients of all parameters in the model.

        Args:
            model (torch.nn.Module): The neural network model.
            norm_type (float): The type of norm to compute (default is 2, which is the L2 norm).

        Returns:
            float: The computed norm of the gradients.
        """
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None and p.requires_grad:
                param_norm = p.grad.detach().data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1.0 / norm_type)

        return total_norm

    def corrcoef(self, x, y):
        y_n = y - y.mean()
        x_n = x - x.mean()
        y_n = y_n / y_n.norm()
        x_n = x_n / x_n.norm()

        return (y_n * x_n).sum()

    def spearman(self, x, y):
        """
        Calculates Spearman's rank correlation coefficient between two tensors.

        Args:
            x (torch.Tensor): First tensor.
            y (torch.Tensor): Second tensor.
            dim (int): Dimension along which to compute the correlation. Default: 0.

        Returns:
            float: Spearman's rank correlation coefficient.
        """
        dtype = x.dtype
        x_rank = x.argsort().argsort().to(dtype)
        y_rank = y.argsort().argsort().to(dtype)

        return self.corrcoef(x_rank, y_rank)

    def spearman_diff(self, x, y):
        n = x.shape[0]
        x_rank = soft_rank(
            x.reshape(1, -1),
            regularization=self.regularization,
            regularization_strength=self.regularization_strength,
        )
        y_rank = soft_rank(
            y.reshape(1, -1),
            regularization=self.regularization,
            regularization_strength=self.regularization_strength,
        )

        return self.corrcoef(x_rank / n, y_rank / n)

    def mse(self, x, y):
        return F.mse_loss(x, y)
