import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Transformer(nn.Module):
    """
    Regression Transformer base model
    """

    def __init__(
        self,
        batch_first: bool = True,
        dim_val: int = 768,
        n_encoder_layers: int = 4,
        n_heads: int = 8,
        dropout_encoder: float = 0.1,
    ) -> None:
        """
        Regression Transformer base model

        Example:
            >>> model = Transformer()
            >>> x = torch.randn(1, 100, 768)
            >>> y = model(x)
            >>> y.shape
            torch.Size([1, 1])
        """
        dim_feedforward_encoder = dim_val * 4

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_encoder_layers, norm=None
        )

        self.reg = nn.Parameter(torch.randn(1, dim_val))
        self.fc1 = nn.Linear(dim_val, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, dim_val)

        Returns:
            Tensor: Output tensor of shape (batch_size, 1)
        """
        x = torch.cat([self.reg.repeat(x.shape[0], 1, 1), x], dim=1)
        x = self.encoder(src=x)
        x = x[:, 0]
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class TransformerS(nn.Module):
    """
    Classification Transformer base model.
    Includes batch normalization layers for faster convergence
    """

    def __init__(
        self,
        batch_first: bool = True,
        dim_val: int = 768,
        n_encoder_layers: int = 8,
        n_heads: int = 8,
        dropout_encoder: float = 0.1,
    ) -> None:
        """
        Classification Transformer base model.
        Includes batch normalization layers for faster convergence

        Note: In the train mode, the model expects batch size of 2 or more. Ths is because of the batch normalization layers.

        Example:
            >>> model = TransformerS()
            >>> x = torch.randn(2, 100, 768)
            >>> y = model(x)
            >>> y.shape
            torch.Size([2, 1])
        """
        dim_feedforward_encoder = dim_val * 4

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_encoder_layers, norm=None
        )

        self.reg = nn.Parameter(torch.randn(1, dim_val))
        self.fc1 = nn.Linear(dim_val, 64)
        self.fc2 = nn.Linear(64, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, dim_val)

        Returns:
            Tensor: Output tensor of shape (batch_size, 1)
        """
        x = torch.cat([self.reg.repeat(x.shape[0], 1, 1), x], dim=1)
        x = self.encoder(src=x)
        x = x[:, 0]

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.gelu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        return x
