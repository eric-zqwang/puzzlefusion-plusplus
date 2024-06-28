import torch.nn as nn
from diffusers.models.attention import Attention, FeedForward
import torch

class MyAdaLayerNorm(nn.Module):
    """
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = emb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x



class EncoderLayer(nn.Module):
    def __init__(self, 
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: int = None,
        attention_bias: bool = False,
        norm_elementwise_affine: bool = True,
        final_dropout: bool = False,
    ):
        super().__init__()

        #  1. self attention
        self.norm1 = MyAdaLayerNorm(dim, num_embeds_ada_norm)  
        
        self.self_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )

        # 2. global attention
        self.norm2 = MyAdaLayerNorm(dim, num_embeds_ada_norm)
        
        self.global_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )

        # 3. feed forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(
            dim, 
            dropout=dropout, 
            activation_fn=activation_fn, 
            final_dropout=final_dropout
        )


    def forward(self, hidden_states, self_mask, gen_mask, timestep):
        # we use ada_layer_norm
        # 1. self attention
        norm_hidden_states = self.norm1(hidden_states, timestep)
        attn_output = self.self_attn(norm_hidden_states, attention_mask=self_mask)
        hidden_states = hidden_states + attn_output

        # 2. global attention
        norm_hidden_states = self.norm2(hidden_states, timestep)
        global_out = self.global_attn(norm_hidden_states, attention_mask=gen_mask)
        hidden_states = hidden_states + global_out 

        # 3. feed forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        return hidden_states