import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .edge_point_process import  get_region_index, point_sample, CrossAttention,CrossAttention_Encoder
from models import register
from .mmseg.models.sam import ImageEncoderViT, MaskDecoder, TwoWayTransformer

logger = logging.getLogger(__name__)
from typing import Any, Optional, Tuple
from torch import nn, Tensor

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 128, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W
class MaskAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cross_attention = CrossAttention(dim, num_heads=4)
        self.fg_num_points = 32
        self.bg_num_points = 32


    def forward(self, x, mask):
        b, c, h ,w =x.size()
        mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
        mask = torch.sigmoid(mask)

        region_index = get_region_index(mask, num_points=self.fg_num_points)
        fg_features = point_sample(x, region_index).permute(0, 2, 1)
        region_index = get_region_index(1 - mask, num_points=self.bg_num_points)
        bg_features = point_sample(x, region_index).permute(0, 2, 1)
        kv_features = torch.cat([fg_features,bg_features], dim=1)
        q_features = x.flatten(2).permute(0, 2, 1)
        cross_attention_features = self.cross_attention(q_features, kv_features, kv_features)
        out_features = cross_attention_features.permute(0, 2, 1).view(b, c, h, w)

        return out_features
class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=16, embed_dim=128):
        super().__init__()

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
        ])
        self.conv1 = nn.Sequential(*[
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(inplanes, inplanes * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes * 2),
            nn.ReLU(inplace=True)
        ])
        self.conv2 = nn.Sequential(*[
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(inplanes * 2, inplanes * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes * 4),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(inplanes * 4, inplanes * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes * 4),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(inplanes * 4, inplanes * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes * 4),
            nn.ReLU(inplace=True)
        ])
        self.convs=nn.ModuleList()

        self.convs.append(nn.Conv2d(inplanes * 2, inplanes * 2, kernel_size=1, stride=1, padding=0, bias=True))
        self.convs.append(nn.Conv2d(inplanes * 4, inplanes * 4, kernel_size=1, stride=1, padding=0, bias=True))
        self.convs.append(nn.Conv2d(inplanes * 4, inplanes * 4, kernel_size=1, stride=1, padding=0, bias=True))
        self.convs.append(nn.Conv2d(inplanes * 4, inplanes * 4, kernel_size=1, stride=1, padding=0, bias=True))

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(14 * inplanes, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU())

    def forward(self, x):
        c0 = self.stem(x)
        c1 = self.conv1(c0)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)


        B, C, H, W = c4.size()
        inputs = [c1,c2,c3,c4]


        out = torch.cat([nn.functional.adaptive_avg_pool2d(self.convs[i](inputs[i]), (H, W)) for i in range(len(inputs))], dim=1)

        out = self.fuse_conv(out)

        return out

class ADDM(nn.Module):
    def __init__(self, inplanes=256):
        super().__init__()

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1),
            LayerNorm2d(inplanes),
            nn.GELU()
        )

    def forward(self, x, ms):
        ms_up = F.interpolate(ms, size=x.size()[2:], mode='bilinear', align_corners=True)

        out = x + ms_up

        return self.fuse_conv(out)
class MSGM(nn.Module):
    def __init__(self, inplanes=128):
        super().__init__()

        self.num_head = 4
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.conv7x7_1 = nn.Sequential(
            LayerNorm2d(self.num_head),
            nn.Conv2d(self.num_head, self.num_head, kernel_size=7, padding=3, bias=False),
        )
        self.conv7x7_2 = nn.Sequential(
            LayerNorm2d(self.num_head),
            nn.Conv2d(self.num_head, self.num_head, kernel_size=7, padding=3, bias=False),
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1),
            LayerNorm2d(inplanes),
            nn.GELU()
        )
        self.ln1 = LayerNorm2d(inplanes)
        self.ln2 = LayerNorm2d(inplanes)
        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1),
            LayerNorm2d(inplanes),
            nn.GELU()
        )


    def forward(self, x, ms):
        b, c, hx, wx = x.size()
        b, c, hs, ws = ms.size()
        nx = hx * wx
        ns = hs * ws

        dx = nn.functional.adaptive_avg_pool2d(x, output_size=(hs, ws))
        fused = ms * dx
        ms = self.fuse_conv1(fused)+ ms


        x_in = x.view(b, self.num_head, c // self.num_head, hx, wx)
        ms_ori = ms.view(b, self.num_head, c // self.num_head, hs, ws)

        ms = self.conv1(self.ln1(ms))
        x = self.conv2(self.ln2(x))
        x_flatten = x.flatten(2).view(b, self.num_head, c // self.num_head, nx)
        ms_flatten = ms.flatten(2).view(b, self.num_head, c // self.num_head, ns)
        attn = (ms_flatten.transpose(-2, -1) @ x_flatten)

        ms_atten = torch.mean(attn, dim=3, keepdim=True).transpose(-2, -1).reshape(b, self.num_head, hs, ws)
        x_atten = torch.mean(attn, dim=2, keepdim=True).reshape(b, self.num_head, hx, wx)

        scale1 = F.sigmoid(self.conv7x7_1(ms_atten)).unsqueeze(2)


        scale2 = F.sigmoid(self.conv7x7_2(x_atten)).unsqueeze(2)


        out1 = scale1 * ms_ori
        out2 = scale2 * x_in
        out1 = out1.reshape(b, c, hs, ws)
        out1 = F.interpolate(out1, size=x.size()[2:], mode='bilinear', align_corners=False)
        out2 = out2.reshape(b, c, hx, wx)
        out = out1 + out2

        return self.fuse_conv(out)



@register('defectsam')
class SAM(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None,down_channel=128):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']

        self.image_encoder = ImageEncoderViT(
            img_size=inp_size,
            patch_size=encoder_mode['patch_size'],
            in_chans=3,
            embed_dim=encoder_mode['embed_dim'],
            depth=encoder_mode['depth'],
            num_heads=encoder_mode['num_heads'],
            mlp_ratio=encoder_mode['mlp_ratio'],
            out_chans=encoder_mode['out_chans'],
            qkv_bias=encoder_mode['qkv_bias'],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=encoder_mode['use_rel_pos'],
            rel_pos_zero_init=True,
            window_size=encoder_mode['window_size'],
            global_attn_indexes=encoder_mode['global_attn_indexes'],
        )


        # self.image_encoder = load_checkpoint(self.image_encoder,state_dict['model'])
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )


        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim']//2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])


        self.neck2 = nn.Sequential(
            nn.Conv2d(768, down_channel, kernel_size=1, bias=False),
            LayerNorm2d(down_channel),
            nn.GELU(),
            nn.Conv2d(down_channel, down_channel, kernel_size=3, padding=1, bias=False, ),
            LayerNorm2d(down_channel),
            nn.GELU(),
        )
        self.neck3 = nn.Sequential(
            nn.Conv2d(768, down_channel, kernel_size=1, bias=False),
            LayerNorm2d(down_channel),
            nn.GELU(),
            nn.Conv2d(down_channel, down_channel, kernel_size=3, padding=1, bias=False, ),
            LayerNorm2d(down_channel),
            nn.GELU(),
        )
        self.neck4 = nn.Sequential(
            nn.Conv2d(768, down_channel, kernel_size=1, bias=False),
            LayerNorm2d(down_channel),
            nn.GELU(),
            nn.Conv2d(down_channel, down_channel, kernel_size=3, padding=1, bias=False, ),
            LayerNorm2d(down_channel),
            nn.GELU()
        )


        self.UpCon1x1_1 = nn.Sequential(
            nn.Conv2d(down_channel, 256, kernel_size=3, padding=1, bias=False, ),
            LayerNorm2d(256),
            nn.GELU()
        )
        self.UpCon1x1_2 = nn.Sequential(
            nn.Conv2d(down_channel, 256, kernel_size=3, padding=1, bias=False, ),
            LayerNorm2d(256),
            nn.GELU()
        )
        self.UpCon1x1_3 = nn.Sequential(
            nn.Conv2d(down_channel, 256, kernel_size=3, padding=1, bias=False, ),
            LayerNorm2d(256),
            nn.GELU()
        )

        self.maskattn1 = MaskAttention(dim=down_channel)
        self.maskattn2 = MaskAttention(dim=down_channel)

        self.out_size = 512
        self.spm = SpatialPriorModule(inplanes=16, embed_dim=down_channel)
        self.msgm2 = MSGM(down_channel)
        self.msgm3 = MSGM(down_channel)
        self.msgm4 = MSGM(down_channel)


        self.linearrr1 = nn.Conv2d(down_channel, 1, kernel_size=3, stride=1, padding=1)
        self.linearrr2 = nn.Conv2d(down_channel, 1, kernel_size=3, stride=1, padding=1)
        self.linearrr3 = nn.Conv2d(down_channel, 1, kernel_size=3, stride=1, padding=1)


    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """

        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self, x):

        bs = 1
        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        y = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=True)
        msfeat = self.spm(y)
        features = self.image_encoder(x)

        vit_features2 = self.neck2(features[1].permute(0, 3, 1, 2))
        vit_features3 = self.neck3(features[2].permute(0, 3, 1, 2))
        vit_features4 = self.neck4(features[3].permute(0, 3, 1, 2))


        vit_features4 = self.msgm4(vit_features4, msfeat)
        vit_features3 = self.msgm3(vit_features3, msfeat)
        vit_features2 = self.msgm2(vit_features2, msfeat)

        mask_11 = self.linearrr1(vit_features2)
        mask_12 = self.linearrr2(vit_features3)
        mask_13 = self.linearrr3(vit_features4)

        mask1, iou_predictions1= self.mask_decoder(
            image_embeddings=self.UpCon1x1_1(vit_features4),
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )



        vit_features3 = self.maskattn1(vit_features3,mask1.detach())
        mask2, iou_predictions2 = self.mask_decoder(
            image_embeddings=self.UpCon1x1_2(vit_features3),
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )


        vit_features2 = self.maskattn2(vit_features2, mask2.detach())
        mask3, iou_predictions3 = self.mask_decoder(
            image_embeddings=self.UpCon1x1_3(vit_features2),
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        main_mask=mask1+mask2+mask3

        main_mask = self.postprocess_masks(main_mask, out_size=(self.out_size, self.out_size))
        mask1 = self.postprocess_masks(mask1, out_size=(self.out_size, self.out_size))
        mask2 = self.postprocess_masks(mask2, out_size=(self.out_size, self.out_size))
        mask3 = self.postprocess_masks(mask3, out_size=(self.out_size, self.out_size))


        mask_11 = self.postprocess_masks(mask_11, out_size=(self.out_size, self.out_size))
        mask_12 = self.postprocess_masks(mask_12, out_size=(self.out_size, self.out_size))
        mask_13 = self.postprocess_masks(mask_13, out_size=(self.out_size, self.out_size))



        return  [main_mask, mask3,mask2, mask1, mask_11, mask_12, mask_13]
        # return [mask1]
        # return [main_mask]

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            out_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(masks, out_size, mode="bilinear", align_corners=False)
        return masks
