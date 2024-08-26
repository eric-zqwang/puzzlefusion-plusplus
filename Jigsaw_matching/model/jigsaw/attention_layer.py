import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch import Tensor


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head=8, d_model=512, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head

        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v, bias=False)
        self.fc = nn.Linear(n_head * self.d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        # x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class CrossAttentionLayer(nn.Module):
    """A cross attention layer"""

    def __init__(self, d_in, n_head):
        super(CrossAttentionLayer, self).__init__()
        self.attn = MultiHeadAttention(n_head, d_in, dropout=0)
        self.pos_ffn = PositionwiseFeedForward(d_in, 2 * d_in, dropout=0)

    def f_layer(self, x, y):
        dx, attn_x = self.attn(x, y, y)
        dx = self.pos_ffn(dx)
        return dx

    def forward(self, x):
        dx = self.f_layer(x, x)
        return dx


class LayerNorm1d(nn.BatchNorm1d):
    def forward(self, input: Tensor) -> Tensor:
        return (
            super()
            .forward(input.transpose(1, 2).contiguous())
            .transpose(1, 2)
            .contiguous()
        )


def knn_and_group(x, xyz, k=10, new_xyz=None, batch_x=None, with_xyz=False):
    """x: [N_sum, C]"""
    device = x.device
    N, n_dim = x.shape
    if batch_x is None:
        batch_x = torch.zeros(N, dtype=torch.long, device=device)
    if new_xyz is None:
        new_xyz = xyz
    idx = torch_geometric.nn.knn(x, x, k, batch_x, batch_x)
    idx, msk = torch_geometric.utils.to_dense_batch(
        idx[1], idx[0], fill_value=N, max_num_nodes=k
    )
    x = torch.cat([x, torch.zeros(1, n_dim).to(device)], dim=0)
    idxx = idx.view(-1).long()
    feature = x[idxx, :]
    feature = feature.view(N, k, n_dim)  # [N, k, dim]
    if with_xyz:
        assert new_xyz.is_contiguous()
        xyz = torch.cat([xyz, torch.zeros(1, 3).to(device)], dim=0)
        grouped_xyz = xyz[idxx, :].view(N, k, 3) - new_xyz.unsqueeze(
            1
        )  # [N, k, 3]
        msk = msk.to(torch.float32)
        grouped_xyz = torch.einsum(
            "n s c, n s -> n s c", grouped_xyz, msk
        )  # (m, num_sample, 3)
        return torch.cat([grouped_xyz, feature], dim=-1), idx
    else:
        return feature, idx


class PointTransformerLayer(nn.Module):
    """A single Point transformer module.
    Modified from:
    https://github.com/Pointcept/Pointcept/blob/864b9c4729c39a95c0a4e5a9f4087f5feafa5df4/pointcept/models/point_transformer/point_transformer_seg.py
    """

    def __init__(self, in_feat, out_feat, n_heads=8, nsampmle=16):
        super(PointTransformerLayer, self).__init__()
        self.mid_feat = mid_feat = out_feat
        self.out_feat = out_feat
        self.share_feat = n_heads
        self.n_sample = nsampmle
        self.linear_q = nn.Linear(in_feat, mid_feat)
        self.linear_k = nn.Linear(in_feat, mid_feat)
        self.linear_v = nn.Linear(in_feat, out_feat)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            LayerNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_feat),
        )
        self.linear_w = nn.Sequential(
            LayerNorm1d(mid_feat),
            nn.ReLU(inplace=True),
            nn.Linear(mid_feat, out_feat // n_heads),
            LayerNorm1d(out_feat // n_heads),
            nn.ReLU(inplace=True),
            nn.Linear(out_feat // n_heads, out_feat // n_heads),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, p, x, o) -> torch.Tensor:
        # [n, 3], [n, c], [b]
        o = o.reshape(-1)
        batch_o = torch.tensor(
            [b for b in range(o.shape[0]) for t in range(int(o[b]))],
            dtype=torch.long,
            device=x.device,
        )
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        x_k, idx = knn_and_group(
            x_k, p, k=self.n_sample, batch_x=batch_o, with_xyz=True
        )
        x_v, _ = knn_and_group(
            x_v, p, k=self.n_sample, batch_x=batch_o, with_xyz=False
        )
        # [N, k, C]
        p_r, x_k = x_k[:, :, :3], x_k[:, :, 3:]
        p_r = self.linear_p(p_r)
        r_qk = (
                x_k
                - x_q.unsqueeze(1)
                + einops.reduce(
            p_r, "n ns (i j) -> n ns j", reduction="sum", j=self.mid_feat
        )
        )
        w = self.linear_w(r_qk)
        w = self.softmax(w)
        x = torch.einsum(
            "n t s i, n t i -> n s i",
            einops.rearrange(
                x_v + p_r, "n ns (s i) -> n ns s i", s=self.share_feat
            ),
            w,
        )
        x = einops.rearrange(x, "n s i -> n (s i)")
        return x


if __name__ == "__main__":
    # a minimum test example
    pos = torch.randn(12, 3)
    x = torch.randn(12, 6)
    b = torch.tensor([4, 3, 5], dtype=torch.long).reshape(3, 1)

    pnf_layer = PointTransformerLayer(
        in_feat=6, out_feat=6, n_heads=2, nsampmle=4
    )
    x_pnf = pnf_layer(p=pos, x=x, o=b)
    print(x_pnf.shape)

    cross_attention_layer = CrossAttentionLayer(d_in=6, n_head=2)
    x = torch.randn(3, 4, 6)
    x_ca = cross_attention_layer(x)
    print(x_ca.shape)
