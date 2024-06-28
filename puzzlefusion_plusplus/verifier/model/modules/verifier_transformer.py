
import torch.nn as nn
from utils.model_utils import (
    PositionalEncoding,
)
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class VerifierTransformer(nn.Module):
    def __init__(self, cfg):
        super(VerifierTransformer, self).__init__()
        self.cfg = cfg
        self.model_channels = cfg.model.embed_dim
        self.num_layers = cfg.model.num_layers
        self.num_heads = cfg.model.num_heads

        transformer_encoder_layer = TransformerEncoderLayer(
            d_model=self.model_channels,
            nhead=self.num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )

        self.transformer_encoder = TransformerEncoder(
            transformer_encoder_layer,
            num_layers=self.num_layers,
            enable_nested_tensor=False
        )
        
        self.edge_indices_pe = PositionalEncoding(self.model_channels // 2, max_len=20) # Max 20 nodes

        self.edge_feature_emb = nn.Linear(7, self.model_channels)

        self.mlp_out = nn.Linear(
            self.model_channels, 
            1
        )


    def forward(self, edge_features, edge_indices, mask):
        """
        data:
            - edge_indices: [B, E, 2]
            - edge_features: [B, E, 6, 1]
            - edge_valid: [B, E]
        """

        B, E, _ = edge_indices.shape

        edge_features = self.edge_feature_emb(edge_features)
        pe = self.edge_indices_pe.pe[0]
        edge_indices_pe = pe[edge_indices].reshape(B, E, -1)

        data_emb = edge_indices_pe + edge_features

        mask = mask.to(torch.bool)


        # Transformer encoder
        edge_features = self.transformer_encoder(data_emb, src_key_padding_mask=~mask)

        out = self.mlp_out(edge_features)

        return out