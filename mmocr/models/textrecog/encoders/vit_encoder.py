# Copyright (c) OpenMMLab. All rights reserved.
from mmcls.models import VisionTransformer
import torch
from mmocr.models.builder import ENCODERS


@ENCODERS.register_module()
class ViTEncoder(VisionTransformer):
    """Implement encoder for ViT

    Args:
        n_layers (int): Number of attention layers.
        n_head (int): Number of parallel attention heads.
        d_k (int): Dimension of the key vector.
        d_v (int): Dimension of the value vector.
        d_model (int): Dimension :math:`D_m` of the input from previous model.
        n_position (int): Length of the positional encoding vector. Must be
            greater than ``max_seq_len``.
        d_inner (int): Hidden dimension of feedforward layers.
        dropout (float): Dropout rate.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(
        self,
        arch="b",
        img_size=224,
        patch_size=16,
        out_indices=-1,
        drop_rate=0,
        drop_path_rate=0,
        norm_cfg=dict(type="LN", eps=1e-6),
        final_norm=True,
        output_cls_token=True,
        interpolate_mode="bicubic",
        patch_cfg=dict(),
        layer_cfgs=dict(),
        init_cfg=None,
        **kwargs
    ):
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            output_cls_token=output_cls_token,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            layer_cfgs=layer_cfgs,
            init_cfg=init_cfg,
        )
        self.pos_embed.requires_grad = False
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)[0]
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.drop_after_pos(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

        return x

    # def forward(self, x, img_metas=None):
    #     """
    #     Args:
    #         feat (Tensor): Feature tensor of shape :math:`(N, D_m, H, W)`.
    #         img_metas (dict): A dict that contains meta information of input
    #             images. Preferably with the key ``valid_ratio``.

    #     Returns:
    #         Tensor: A tensor of shape :math:`(N, T, D_m)`.
    #     """
    #     valid_ratios = [1.0 for _ in range(feat.size(0))]
    #     if img_metas is not None:
    #         valid_ratios = [img_meta.get("valid_ratio", 1.0) for img_meta in img_metas]
    #     feat += self.position_enc(feat)
    #     n, c, h, w = feat.size()
    #     mask = feat.new_zeros((n, h, w))
    #     for i, valid_ratio in enumerate(valid_ratios):
    #         valid_width = min(w, math.ceil(w * valid_ratio))
    #         mask[i, :, :valid_width] = 1
    #     mask = mask.view(n, h * w)
    #     feat = feat.view(n, c, h * w)

    #     output = feat.permute(0, 2, 1).contiguous()
    #     for enc_layer in self.layer_stack:
    #         output = enc_layer(output, h, w, mask)
    #     output = self.layer_norm(output)

    #     return output
