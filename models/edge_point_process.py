import torch
import warnings
import math
def select_from_topK(uncertainty_map, num_points):
    R, _, H, W = uncertainty_map.shape
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    return point_indices


def get_region_index(mask, num_points):
    R, _, H, W = mask.shape

    point_indices = select_from_topK(mask, num_points)

    return point_indices


def point_sample(input, point_indices, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_indices (Tensor): A tensor of shape (N, P) or (N, Hgrid, Wgrid, 2) that contains sampled indices.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    N, C, H, W = input.shape
    point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
    flatten_input = input.flatten(start_dim=2)
    sampled_feats = flatten_input.gather(dim=2, index=point_indices).view_as(point_indices)
    return sampled_feats
import torch.nn as nn



class CrossAttention_Encoder(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim

        self.scale = qk_scale or head_dim ** -0.5

        self.Q = nn.Linear(dim, dim, bias=qkv_bias)
        self.K = nn.Linear(dim, dim, bias=qkv_bias)
        self.V = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N_q, C_q = q.shape
        _, N_kv, C_kv = k.shape

        q = self.Q(q).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.K(k).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.V(v).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C_q)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):

    def __init__(self, dim=256, num_heads=4,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm):
        super().__init__()

        self.ca = CrossAttention_Encoder(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm = norm_layer(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, q, k, v):


        q = self.ca(q, k, v)+q

        q = self.norm(q)

        return q