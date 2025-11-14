
import torch
import torch.nn as nn
from pyshred.models.sequence_models.abstract_sequence import AbstractSequence

class ProjectedLSTM(AbstractSequence):
    """LSTM + linear projection that conforms to PySHRED's AbstractSequence API.
    hidden_size_upstream: size of internal LSTM hidden state.
    proj_dim: latent dimension exposed to SHRED/decoder/SINDy.
    num_layers: number of LSTM layers.
    layer_norm: whether to apply layer norm to the LSTM outputs before projection.
    """
    def __init__(self, hidden_size_upstream: int = 64, proj_dim: int = 8,
                 num_layers: int = 2, layer_norm: bool = False):
        super().__init__()
        self.hidden_size_upstream = hidden_size_upstream
        self.proj_dim = proj_dim
        self.num_layers = num_layers
        self.use_layer_norm = layer_norm
        self.lstm = None          # lazy init
        self.decoder_type = None  # set during initialize
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.hidden_size_upstream)
        # Expose hidden_size = projected dimension so SHRED treats proj_dim as latent dimension
        self.hidden_size = proj_dim
        self.output_size = proj_dim
        self.proj = nn.Linear(self.hidden_size_upstream, self.proj_dim)

    def initialize(self, input_size: int, decoder_type: str, **kwargs):
        super().initialize(input_size)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size_upstream,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.decoder_type = decoder_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        super().forward(x)
        device = next(self.parameters()).device
        h0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size_upstream), device=device)
        c0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size_upstream), device=device)
        out, (h_out, c_out) = self.lstm(x, (h0, c0))
        if self.use_layer_norm:
            out = self.layer_norm(out)
            last = self.layer_norm(h_out[-1])
        else:
            last = h_out[-1]
        # Apply projection
        last_proj = self.proj(last)            # (batch, proj_dim)
        out_proj = self.proj(out)              # (batch, seq_len, proj_dim)
        if self.decoder_type == "MLP":
            return last_proj
        elif self.decoder_type == "UNET":
            # UNET expects (batch, channels, length)
            return out_proj.permute(0, 2, 1)
        else:
            raise TypeError(f"Unsupported decoder type: {self.decoder_type}")

    @property
    def model_name(self) -> str:
        return "ProjectedLSTM"
