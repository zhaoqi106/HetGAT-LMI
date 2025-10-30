# hetgat_lmi_module.py
# Pure model module for lnc–mi interaction prediction.

from __future__ import annotations
from typing import Dict, Tuple
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, HeteroConv


class GateLayer(nn.Module):
    """Pairwise gated fusion: g*(f_lnc) + (1-g)*(f_mi)."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Sigmoid(),
        )

    def forward(self, f_lnc: torch.Tensor, f_mi: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([f_lnc, f_mi], dim=1))
        return g * f_lnc + (1.0 - g) * f_mi


class HetGNNLayer(nn.Module):
    """Relation-specific GATv2 over a heterogeneous graph."""
    def __init__(self, in_ch: Dict[str, int], out_dim: int, heads: int = 8):
        super().__init__()
        self.conv = HeteroConv(
            {
                ("lnc", "similar", "lnc"): GATv2Conv(
                    in_ch["lnc"], out_dim, heads=heads, add_self_loops=False
                ),
                ("mi", "similar", "mi"): GATv2Conv(
                    in_ch["mi"], out_dim, heads=heads, add_self_loops=False
                ),
                ("lnc", "interacts", "mi"): GATv2Conv(
                    (in_ch["lnc"], in_ch["mi"]), out_dim, heads=heads, add_self_loops=False
                ),
            },
            aggr="mean",
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return self.conv(x_dict, edge_index_dict)


class HetGNN(nn.Module):
    """
    Heterogeneous GATv2 model with pairwise gated fusion and MLP classifier.

    Forward inputs:
      - x_dict: {"lnc": [N_lnc, F_l], "mi": [N_mi, F_m]}
      - edge_index_dict: {("lnc","similar","lnc"), ("mi","similar","mi"), ("lnc","interacts","mi")}
      - pairs: LongTensor [B, 2] with columns [lnc_index, mi_index]
    """
    def __init__(
        self,
        in_ch: Dict[str, int],
        hidden_dim: int = 128,
        heads: int = 8,
        dropout_gate: float = 0.3,
        clf_drop1: float = 0.3,
        clf_drop2: float = 0.2,
    ):
        super().__init__()
        self.encoder = HetGNNLayer(in_ch, hidden_dim, heads=heads)
        feat_dim = hidden_dim * heads

        self.gate_layer = GateLayer(feat_dim)
        self.dropout_gate = nn.Dropout(dropout_gate)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(clf_drop1),
            nn.Linear(feat_dim // 2, feat_dim // 4),
            nn.ReLU(),
            nn.Dropout(clf_drop2),
            nn.Linear(feat_dim // 4, 1),
        )

    # ---- utilities for analysis/inference ----
    def encode(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return relation-encoded node embeddings per type."""
        return self.encoder(x_dict, edge_index_dict)

    def pair_embed(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        """Return fused pair embeddings before the classifier."""
        x = self.encode(x_dict, edge_index_dict)
        src = pairs[:, 0].long()
        tgt = pairs[:, 1].long()
        f_lnc = x["lnc"][src]
        f_mi = x["mi"][tgt]
        fused = self.gate_layer(f_lnc, f_mi)
        return self.dropout_gate(fused)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits for each lnc–mi pair."""
        fused = self.pair_embed(x_dict, edge_index_dict, pairs)
        return self.classifier(fused).squeeze(1)

    @torch.no_grad()
    def predict_proba(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        """Return probabilities in [0,1] for each pair."""
        return torch.sigmoid(self.forward(x_dict, edge_index_dict, pairs))

