"""GraphSAGE encoder with an edge head.

The encoder gives every Account an embedding from its snapshot neighbourhood.
With `gnn.encoder: edge_sage`, messages carry the neighbour state and the
Transaction attributes, and forward and reverse edges use separate weights.
With `gnn.encoder: sage`, the layers are PyG `SAGEConv` and messages use the
neighbour states only. The edge head scores one Transaction from the
embeddings of its two Accounts and its own attributes in both cases.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import MessagePassing, SAGEConv

from sage_xgb.config import GnnConfig
from sage_xgb.data import vocab
from sage_xgb.gnn.graph import (
    CATEGORY_START,
    CONTINUOUS_COLUMNS,
    DIRECTION_COLUMN,
    NODE_FEATURE_DIM,
)


_CATEGORY_DIM = 4


class EdgeSageConv(MessagePassing):
    """One SAGE layer whose messages are conditioned on the edge encoding."""

    def __init__(self, hidden: int) -> None:
        """Create the layer with `hidden` input and output channels."""
        super().__init__(aggr="mean")
        self.forward_message = nn.Linear(2 * hidden, hidden)
        self.reverse_message = nn.Linear(2 * hidden, hidden)
        self.self_update = nn.Linear(hidden, hidden)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_hidden: torch.Tensor,
        is_reverse: torch.Tensor,
    ) -> torch.Tensor:
        """Return the updated node states."""
        aggregated = self.propagate(
            edge_index, x=x, edge_hidden=edge_hidden, is_reverse=is_reverse
        )
        return self.self_update(x) + aggregated

    def message(
        self,
        x_j: torch.Tensor,
        edge_hidden: torch.Tensor,
        is_reverse: torch.Tensor,
    ) -> torch.Tensor:
        """Return one message per edge, with separate weights per direction."""
        joined = torch.cat([x_j, edge_hidden], dim=-1)
        return torch.where(
            is_reverse.unsqueeze(-1),
            self.reverse_message(joined),
            self.forward_message(joined),
        )


class GnnModel(nn.Module):
    """Account encoder plus Transaction head.

    `encode_nodes` works on any sampled or full subgraph. `score_edges` turns
    two Account embeddings and a Transaction's attributes into a logit and an
    embedding of `embedding_dim` values.
    """

    def __init__(self, config: GnnConfig) -> None:
        """Create the model from the GNN settings."""
        super().__init__()
        hidden = config.hidden
        self.node_in = nn.Linear(NODE_FEATURE_DIM, hidden)
        self.category_embeddings = nn.ModuleList(
            [
                nn.Embedding(len(vocab.CURRENCIES), _CATEGORY_DIM),
                nn.Embedding(len(vocab.CURRENCIES), _CATEGORY_DIM),
                nn.Embedding(len(vocab.PAYMENT_FORMATS), _CATEGORY_DIM),
            ]
        )
        edge_in = CONTINUOUS_COLUMNS + 1 + _CATEGORY_DIM * len(self.category_embeddings)
        self.edge_in = nn.Linear(edge_in, hidden)
        self.plain_sage = config.encoder == "sage"
        layers: list[nn.Module] = []
        for _ in range(config.layers):
            layers.append(
                SAGEConv(hidden, hidden) if self.plain_sage else EdgeSageConv(hidden)
            )
        self.convs = nn.ModuleList(layers)
        self.head_hidden = nn.Linear(3 * hidden, config.embedding_dim)
        self.head_out = nn.Linear(config.embedding_dim, 1)

    def encode_edges(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Return the hidden encoding of Transaction attributes."""
        codes = edge_attr[:, CATEGORY_START:].long()
        embedded = [
            table(codes[:, column])
            for column, table in enumerate(self.category_embeddings)
        ]
        direction = edge_attr[:, DIRECTION_COLUMN : DIRECTION_COLUMN + 1]
        continuous = edge_attr[:, :CONTINUOUS_COLUMNS]
        return torch.relu(
            self.edge_in(torch.cat([continuous, direction, *embedded], -1))
        )

    def encode_nodes(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Return one embedding per node of the (sub)graph."""
        h = torch.relu(self.node_in(x))
        if self.plain_sage:
            for conv in self.convs:
                h = torch.relu(conv(h, edge_index))
            return h
        if edge_index.numel() == 0:
            edge_hidden = h.new_zeros((0, h.shape[1]))
            is_reverse = torch.zeros(0, dtype=torch.bool)
        else:
            edge_hidden = self.encode_edges(edge_attr)
            is_reverse = edge_attr[:, DIRECTION_COLUMN] > 0.5
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index, edge_hidden, is_reverse))
        return h

    def score_edges(
        self, h_src: torch.Tensor, h_dst: torch.Tensor, edge_attr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the logit and the embedding of each Transaction."""
        joined = torch.cat([h_src, h_dst, self.encode_edges(edge_attr)], dim=-1)
        embedding = torch.relu(self.head_hidden(joined))
        return self.head_out(embedding).squeeze(-1), embedding
