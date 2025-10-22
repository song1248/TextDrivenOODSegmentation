import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from mmseg.models.backbones import ResNet
from mmseg.models.builder import BACKBONES
from timm.models.layers import drop_path, trunc_normal_

import copy


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class ResidualAugmentedAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    # def attention(self, x: torch.Tensor, seg_mask=None):
    #     self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
    #     # Extract unique values from seg_mask
    #     unique_values = torch.unique(seg_mask)
    #     value = 19
    #     mask = (seg_mask == value)
    #     assert False
    #     masked_attn = self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    #     original_attn = self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    #     return ???
    #     # return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    def attention(self, x: torch.Tensor, seg_mask=None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None

        if seg_mask is not None: # train : segmask가 들어옴  (inference 시에는 segmask가 안들어옴)

            seq_length, batch_size, embedding_dim = x.shape
            _, _, height, width = seg_mask.shape

            batch_size, _, height, width = seg_mask.shape

            seg_mask = (seg_mask == 19).float()

            # Flatten and interpolate seg_mask to match sequence length
            seg_mask = torch.nn.functional.interpolate(
                seg_mask, size=(int((seq_length-1) ** 0.5), int((seq_length-1) ** 0.5)), mode='bilinear', align_corners=False
            ).view(batch_size, -1)

            # # 일부는 마스크 안하기 (20%) # method 3.5
            # if True:
            #     # Define the probability for flipping 1 to 0
            #     probability = 0.2  # 20% of 1s will be flipped to 0

            #     # Generate a random tensor of the same shape as seg_mask
            #     random_tensor = torch.rand_like(seg_mask)

            #     # Create a mask where random values are less than the probability and seg_mask is 1
            #     seg_mask = (random_tensor < probability) & (seg_mask == 1)
            attn_mask_inside = seg_mask
            attn_mask_outside = 1.0 - seg_mask

            # 추가할 값 (1.0)을 같은 배치 크기로 생성
            cls_one_mask = torch.ones(batch_size, 1).to(seg_mask)  # (4, 1)
            cls_zero_mask = torch.zeros(batch_size, 1).to(seg_mask)  # (4, 1)

            # 기존 텐서 앞에 추가
            attn_mask_inside = torch.cat([cls_zero_mask, attn_mask_inside], dim=1).T.unsqueeze(-1)  # (4, 1025)
            attn_mask_outside = torch.cat([cls_one_mask, attn_mask_outside], dim=1).T.unsqueeze(-1)  # (4, 1025)

            cls_token = x[:1, :, :]  # CLS 토큰 (1, embedding_dim, seq_length)

            # 채널별 평균과 분산 계산
            mean_per_channel = cls_token.mean(dim=-1, keepdim=True)  # (1, embedding_dim, 1)
            std_per_channel = cls_token.std(dim=-1, keepdim=True)    # (1, embedding_dim, 1)

            # 가우시안 노이즈 생성 (채널별로 크기 조정)
            alpha = 1.0  # 노이즈 강도 조절
            noise = torch.normal(mean=0.0, std=1.0, size=cls_token.size()).to(cls_token.device)  # 기본 노이즈 생성
            adaptive_noise = alpha * noise * std_per_channel + mean_per_channel  # 채널별로 조정된 노이즈

            # # 50% 확률로 노이즈 추가
            if torch.rand(1).item() < 0.5:  # 50% 확률
                cls_token_noised = cls_token + adaptive_noise
            else:
                cls_token_noised = cls_token  # 노이즈 추가하지 않음

            # 나머지 시퀀스와 결합
            x_copy = torch.cat([cls_token_noised, x[1:, :, :]])

            # 방법 1
                # q k, v에 노이즈
            # 방법 2
                # 2-1 : q에만  노이즈, k, v에는 노이즈를 주지 않음
                # 2-2 : q, k 에 노이즈, v에는 노이즈를 주지 않음
            # 방법 3
                # 3-1 : q의 일부에만 노이즈(ood객체의 일부에만 노이즈)
                # 3-2 : 확률적으로 적용 혹은 적용 안하기

            # Apply the attention masks
            noise_attn = self.attn(x_copy, x, x, need_weights=False, attn_mask=self.attn_mask)[0]  #query에 노이즈 줘야함
            original_attn = self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

            # mask 적용
            masked_attn = noise_attn * attn_mask_inside # 안쪽이 1 바깥이 0
            original_attn = original_attn * attn_mask_outside # 안쪽이 0 바깥이 1


            # Combine the results (example: weighted sum)
            combined_attn = masked_attn + original_attn
        else: # inference : inference 시에는 segmask가 안들어옴
            combined_attn = self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

        return combined_attn

    def forward(self, x: torch.Tensor, H=None, W=None, seg_mask=None):
        x = x + self.drop_path(self.attention(self.ln_1(x), seg_mask))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x



class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, H=None, W=None, seg_mask=None):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x

########## original
class Transformer(nn.Module):

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule

        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, dpr[i]) for i in range(layers)])


    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class Custom_vision_Transformer(nn.Module):

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate=0.):
        """
        Transformer module that selectively uses ResidualAugmentedAttentionBlock.

        Args:
            width (int): Width of the model.
            layers (int): Number of layers in the transformer.
            heads (int): Number of attention heads.
            attn_mask (torch.Tensor, optional): Attention mask.
            drop_path_rate (float): Drop path rate.
            augmented_layers (list of int, optional): Indices of layers to use ResidualAugmentedAttentionBlock.
        """
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule

        augmented_layers = [11]
        # augmented_layers = None
        if augmented_layers is None:
            augmented_layers = []  # Default to an empty list if no layers are specified

        self.resblocks = nn.Sequential(
            *[ResidualAugmentedAttentionBlock(width, heads, attn_mask, dpr[i]) if i in augmented_layers else
              ResidualAttentionBlock(width, heads, attn_mask, dpr[i]) for i in range(layers)]
        )

    def forward(self, x: torch.Tensor, seg_mask=None):
        return self.resblocks(x, seg_mask)



class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x

@BACKBONES.register_module()
class CLIPVisionTransformer(nn.Module):

    def __init__(self,
                 input_resolution=224,
                 patch_size=32,
                 width=768,
                 layers=12,
                 heads=12,
                 output_dim=512,
                 drop_path_rate=0.0,
                 out_indices=[3, 5, 7, 11],
                 pretrained=None,
                 get_embeddings=False,
                 ignore_last_attn=False,
                 **kwargs):

        super().__init__()

        self.embed_dim = width
        self.output_dim = output_dim
        self.pretrained = pretrained

        if isinstance(input_resolution, int):
            self.input_resolution = (input_resolution, input_resolution)
        elif isinstance(input_resolution, tuple):
            self.input_resolution = input_resolution

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn((self.input_resolution[0] // patch_size) * (self.input_resolution[1] // patch_size) + 1, width))
        self.spatial_size = (self.input_resolution[0] // patch_size, self.input_resolution[1] // patch_size)
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.ln_pre = LayerNorm(width)
        self.get_embeddings = get_embeddings

        ####### original
        # self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)

        self.transformer = Custom_vision_Transformer(width, layers, heads, drop_path_rate=drop_path_rate)


        self.out_indices = out_indices
        self.ignore_last_attn = ignore_last_attn

        if get_embeddings:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.fpn_dim = width
        self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(self.fpn_dim),
                nn.GELU(),
                nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2))
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if pretrained:
            warnings.warn(f'Using backbone weights: {pretrained}', stacklevel=2)
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            if 'positional_embedding' in state_dict.keys():
                if self.positional_embedding.shape != state_dict['positional_embedding'].shape:
                    warnings.warn(
                        f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} '
                        f'to {self.positional_embedding.shape}', stacklevel=2)
                    cls_pos = state_dict["positional_embedding"][0:1, :]
                    spatial_pos = F.interpolate(state_dict["positional_embedding"][1:,].reshape(1, 14, 14, 768).permute(0, 3, 1, 2), size=self.spatial_size, mode='bilinear')
                    spatial_pos = spatial_pos.reshape(768, self.spatial_size[0]*self.spatial_size[1]).permute(1, 0)
                    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                    state_dict['positional_embedding'] = positional_embedding
                    assert self.positional_embedding.shape == state_dict['positional_embedding'].shape

            u, w = self.load_state_dict(state_dict, False)
            warnings.warn(f'{u}, {w} are misaligned params in vision transformer', stacklevel=2)

    def forward(self, x: torch.Tensor, seg_mask=None):
        x = self.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0,:] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[1:,].reshape(1, self.spatial_size[0], self.spatial_size[1], C).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H*W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)

        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)

        features = []
        for i, blk in enumerate(self.transformer.resblocks):
            if self.ignore_last_attn:
                mask = torch.empty(x.shape[0], x.shape[0])
                mask.fill_(float('-inf'))
                mask.fill_diagonal_(0)
                self.transformer.resblocks[-1].attn_mask = mask
            x = blk(x, seg_mask = seg_mask)
            if i in self.out_indices:
                xp = x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W)
                features.append(xp.contiguous())

        # FPN original

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])


        # rein FPN 정현
        # features[0] = F.interpolate(
        #     features[0], scale_factor=4, mode="bilinear", align_corners=False
        # )
        # features[1] = F.interpolate(
        #     features[1], scale_factor=2, mode="bilinear", align_corners=False
        # )
        # features[3] = F.interpolate(
        #     features[3], scale_factor=0.5, mode="bilinear", align_corners=False
        # )

        if self.get_embeddings:
            x = x.permute(1, 0, 2)
            x = self.ln_post(x)
            x = x @ self.proj

            global_embedding = x[:, :1]
            visual_embedding = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2) # B C H W

            features.append([global_embedding, visual_embedding])

        return tuple(features)

@BACKBONES.register_module()
class CLIPTextEncoder(nn.Module):

    def __init__(self, context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=1024,
                 out_dim=256,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k]

                if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        warnings.warn(
                            f'positional_embedding is truncated from 77 to {self.context_length}',
                            stacklevel=2)
                    state_dict[k] = checkpoint[k]

            u, w = self.load_state_dict(state_dict, False)
            warnings.warn(f'{u}, {w} are misaligned params in text encoder', stacklevel=2)


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        # x = self.out_proj(x)
        return x

@BACKBONES.register_module()
class CLIPTextContextEncoder(nn.Module):
    def __init__(self, context_length=22,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=1024,
                 out_dim=256,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.embed_dim = embed_dim

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if pretrained:
            warnings.warn(f'Using text encoder weights: {pretrained}', stacklevel=2)
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k]

                if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        warnings.warn(
                            f'positional_embedding is truncated from 77 to {self.context_length}',
                            stacklevel=2)
                    state_dict[k] = checkpoint[k]

            u, w = self.load_state_dict(state_dict, False)
            warnings.warn(f'{u}, {w} are misaligned params in text encoder', stacklevel=2)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask


    def forward(self, text, context=None, learnable_text=None):
        """
        Forward method for CLIPTextContextEncoder.

        Args:
            text: torch.Size([20, 5])  # 기존 클래스 텍스트 토큰
            context: torch.Size([1, 5, 512]) 또는 [B, 5, 512]  # 컨텍스트 임베딩
            learnable_text: torch.Size([M, 5, 512])  # 추가 클래스 텍스트 임베딩

        Returns:
            x: torch.Tensor of shape [B, K, embed_dim]  # 최종 출력
        """
        if context is not None:
            # 1. 기존 텍스트 임베딩
            x_text = self.token_embedding(text)  # (K, N1, C) = (20, 5, 512)
            # 2. learnable_text 결합
            if learnable_text is not None:
                # x_text와 learnable_text 결합
                x_text = torch.cat([x_text, learnable_text], dim=0)  # (K + M, N1, C) = (24, 5, 512)

                # text 확장: learnable_text에 대응하는 더미 텍스트 생성
                learnable_text_tokens = torch.full((learnable_text.shape[0], text.shape[1]), 0, dtype=text.dtype, device=text.device)
                text = torch.cat([text, learnable_text_tokens], dim=0)  # (K + M, N1) = (24, 5)

            # 3. EOS 인덱스 계산
            eos_indx = text.argmax(dim=-1)  # (K + M) = (24)

            # 4. 텍스트 크기와 컨텍스트 크기 일치화
            K, N1, C = x_text.shape  # K = 24
            if len(context.shape) == 3:  # [B, N2, C]
                B, N2, C2 = context.shape

                eos_indx = eos_indx.reshape(1, K).expand(B, K).reshape(-1)  # [B*K]
                x_text = x_text.reshape(1, K, N1, C).expand(B, K, N1, C)  # [B, K, N1, C]
                context = context.reshape(B, 1, N2, C).expand(B, K, N2, C)  # [B, K, N2, C]
            elif len(context.shape) == 4:  # [B, K, N2, C]
                B, K_ctx, N2, C2 = context.shape

                eos_indx = eos_indx.reshape(1, K).expand(B, K).reshape(-1)  # [B*K]
                x_text = x_text.reshape(1, K, N1, C).expand(B, K, N1, C)  # [B, K, N1, C]

            # 5. 컨텍스트 결합 및 Transformer 입력 준비
            x = torch.cat([x_text[:, :, 0:1], context, x_text[:, :, 1:]], dim=2)  # [B, K, N1+N2, C]
            x = x.reshape(B*K, N1 + N2, C)  # [B*K, N1+N2, C]

            # 6. Transformer 처리
            x = x.permute(1, 0, 2)  # [L, N, C]
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # [N, L, C]

            # 7. 최종 LayerNorm 및 투영
            x = self.ln_final(x)
            x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection  # [N, embed_dim]
            x = x.reshape(B, K, self.embed_dim)  # [B, K, embed_dim]

            return x

        else:
            # context가 없는 경우 (원본 로직 그대로)
            x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
            return x



@BACKBONES.register_module()
class ContextDecoder(nn.Module):
    def __init__(self,
                 transformer_width=256,
                 transformer_heads=4,
                 transformer_layers=6,
                 visual_dim=1024,
                 dropout=0.1,
                 **kwargs):
        super().__init__()

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )

        self.decoder = nn.ModuleList([
                    TransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
                ])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, text, visual):
        B, N, C = visual.shape
        visual = self.memory_proj(visual)
        x = self.text_proj(text)

        for layer in self.decoder:
            x = layer(x, visual)

        return self.out_proj(x)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# class AttentionPool2d(nn.Module):
#     def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
#         super().__init__()
#         self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
#         self.num_heads = num_heads
#         self.embed_dim = embed_dim
#         self.spacial_dim = spacial_dim

#     def forward(self, x, seg_mask=None):
#         B, C, H, W = x.shape
#         x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
#         x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

#         cls_pos = self.positional_embedding[0:1, :]
#         spatial_pos = F.interpolate(self.positional_embedding[1:,].reshape(1, self.spacial_dim, self.spacial_dim, self.embed_dim).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
#         spatial_pos = spatial_pos.reshape(self.embed_dim, H*W).permute(1, 0)
#         positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)

#         x = x + positional_embedding[:, None, :]
#         x, _ = F.multi_head_attention_forward(
#             query=x, key=x, value=x,
#             embed_dim_to_check=x.shape[-1],
#             num_heads=self.num_heads,
#             q_proj_weight=self.q_proj.weight,
#             k_proj_weight=self.k_proj.weight,
#             v_proj_weight=self.v_proj.weight,
#             in_proj_weight=None,
#             in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
#             bias_k=None,
#             bias_v=None,
#             add_zero_attn=False,
#             dropout_p=0,
#             out_proj_weight=self.c_proj.weight,
#             out_proj_bias=self.c_proj.bias,
#             use_separate_proj_weight=True,
#             training=self.training,
#             need_weights=False
#         )

#         x = x.permute(1, 2, 0)
#         global_feat = x[:, :, 0]
#         feature_map = x[:, :, 1:].reshape(B, -1, H, W)
#         return global_feat, feature_map



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class AttentionPool2d(nn.Module):
#     def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
#         """
#         Args:
#             spacial_dim (int): 입력 특성 맵의 공간 차원 (예, 7)
#             embed_dim (int): 임베딩 차원 (예, 512)
#             num_heads (int): multi-head attention의 헤드 수
#             output_dim (int, optional): 출력 차원. None이면 embed_dim과 동일.
#         """
#         super().__init__()
#         # CLS 토큰 포함: (spacial_dim**2 + 1, embed_dim)
#         self.positional_embedding = nn.Parameter(
#             torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5
#         )
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
#         self.num_heads = num_heads
#         self.embed_dim = embed_dim
#         self.spacial_dim = spacial_dim

#     def forward(self, x, seg_mask=None):
#         """
#         Args:
#             x (Tensor): 입력 특성 맵, shape = (B, C, H, W)
#             seg_mask (Tensor, optional): (B, 1 또는 다른 채널 수, H, W)
#                 특정 라벨 (예, 19)에 해당하는 영역을 담고 있다고 가정.

#         Returns:
#             global_feat (Tensor): CLS 토큰에서 추출한 전역 특성, shape = (B, C)
#             feature_map (Tensor): 나머지 영역을 복원한 특성 맵, shape = (B, C, H, W)
#         """
#         B, C, H, W = x.shape

#         # (1) NCHW -> (HW, B, C)
#         x = x.reshape(B, C, H * W).permute(2, 0, 1)  # (HW, B, C)

#         # (2) CLS 토큰 추가: 입력의 평균을 CLS 토큰으로 사용 (1, B, C)
#         cls_token = x.mean(dim=0, keepdim=True)
#         x = torch.cat([cls_token, x], dim=0)  # (HW+1, B, C)

#         # (3) positional embedding 추가
#         # CLS 토큰에 해당하는 positional embedding (1, embed_dim)
#         cls_pos = self.positional_embedding[0:1, :]
#         # 나머지 spatial embedding (spacial_dim**2, embed_dim)
#         spatial_pos = self.positional_embedding[1:, :]
#         # 공간 정보를 H, W 크기에 맞게 보간
#         spatial_pos = spatial_pos.reshape(1, self.spacial_dim, self.spacial_dim, self.embed_dim)
#         spatial_pos = spatial_pos.permute(0, 3, 1, 2)  # (1, embed_dim, spacial_dim, spacial_dim)
#         spatial_pos = F.interpolate(spatial_pos, size=(H, W), mode='bilinear', align_corners=False)
#         spatial_pos = spatial_pos.reshape(self.embed_dim, H * W).permute(1, 0)  # (HW, embed_dim)
#         # CLS + spatial 결합 -> (HW+1, embed_dim)
#         positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
#         # 모든 토큰에 positional embedding 추가 (HW+1, B, embed_dim)
#         x = x + positional_embedding[:, None, :]

#         # -----------------------------
#         # seg_mask가 주어진 경우: 노이즈 attention + 원본 attention을 마스크로 결합
#         # -----------------------------
#         if seg_mask is not None:
#             # seg_mask에서 label==19인 영역만 1, 나머지는 0
#             seg_mask = (seg_mask == 19).float()  # (B, 1, H, W) 또는 (B, _, H, W)
#             seg_mask = F.interpolate(seg_mask, size=(H, W), mode='bilinear', align_corners=False)
#             seg_mask = seg_mask.view(B, -1)  # (B, HW)

#             # CLS 토큰에 해당하는 0을 앞에 추가 -> (B, HW+1)
#             cls_zero = torch.zeros(B, 1, device=seg_mask.device)
#             inside_mask = torch.cat([cls_zero, seg_mask], dim=1)
#             outside_mask = 1.0 - inside_mask

#             # (B, HW+1) -> (HW+1, B, 1)
#             inside_mask = inside_mask.transpose(0, 1).unsqueeze(-1)
#             outside_mask = outside_mask.transpose(0, 1).unsqueeze(-1)

#             # -----------------------------
#             # (A) 노이즈를 추가한 Query (CLS 토큰에만)
#             # -----------------------------
#             x_copy = x.clone()
#             cls_token_original = x_copy[:1, :, :]  # (1, B, C)
#             mean_per_channel = cls_token_original.mean(dim=-1, keepdim=True)
#             std_per_channel = cls_token_original.std(dim=-1, keepdim=True)
#             noise = torch.randn_like(cls_token_original)
#             alpha = 0.002
#             adaptive_noise = alpha * noise * std_per_channel + mean_per_channel

#             if torch.rand(1).item() < 0.5:
#                 cls_token_noised = cls_token_original + adaptive_noise
#             else:
#                 cls_token_noised = cls_token_original

#             # in-place 연산 대신 out-of-place 연산 사용
#             x_copy = torch.cat([cls_token_noised, x_copy[1:]], dim=0)

#             # (A1) 노이즈 attention 계산 (Query: 노이즈가 있는 x_copy)
#             noise_attn, _ = F.multi_head_attention_forward(
#                 query=x_copy, key=x, value=x,
#                 embed_dim_to_check=self.embed_dim,
#                 num_heads=self.num_heads,
#                 q_proj_weight=self.q_proj.weight,
#                 k_proj_weight=self.k_proj.weight,
#                 v_proj_weight=self.v_proj.weight,
#                 in_proj_weight=None,
#                 in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
#                 bias_k=None,
#                 bias_v=None,
#                 add_zero_attn=False,
#                 dropout_p=0,
#                 out_proj_weight=self.c_proj.weight,
#                 out_proj_bias=self.c_proj.bias,
#                 use_separate_proj_weight=True,
#                 training=self.training,
#                 need_weights=False
#             )

#             # (A2) 원본 attention 계산 (Query: 원래의 x)
#             original_attn, _ = F.multi_head_attention_forward(
#                 query=x, key=x, value=x,
#                 embed_dim_to_check=self.embed_dim,
#                 num_heads=self.num_heads,
#                 q_proj_weight=self.q_proj.weight,
#                 k_proj_weight=self.k_proj.weight,
#                 v_proj_weight=self.v_proj.weight,
#                 in_proj_weight=None,
#                 in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
#                 bias_k=None,
#                 bias_v=None,
#                 add_zero_attn=False,
#                 dropout_p=0,
#                 out_proj_weight=self.c_proj.weight,
#                 out_proj_bias=self.c_proj.bias,
#                 use_separate_proj_weight=True,
#                 training=self.training,
#                 need_weights=False
#             )

#             # seg_mask (inside: label 19) 에 따라 두 attention 결과를 혼합
#             masked_attn = noise_attn * inside_mask       # label==19 영역은 노이즈 attention 사용
#             original_attn_masked = original_attn * outside_mask  # 그 외 영역은 원본 attention 사용
#             x = masked_attn + original_attn_masked

#         else:
#             # seg_mask가 없는 경우 단순히 원래의 attention 계산
#             x, _ = F.multi_head_attention_forward(
#                 query=x, key=x, value=x,
#                 embed_dim_to_check=self.embed_dim,
#                 num_heads=self.num_heads,
#                 q_proj_weight=self.q_proj.weight,
#                 k_proj_weight=self.k_proj.weight,
#                 v_proj_weight=self.v_proj.weight,
#                 in_proj_weight=None,
#                 in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
#                 bias_k=None,
#                 bias_v=None,
#                 add_zero_attn=False,
#                 dropout_p=0,
#                 out_proj_weight=self.c_proj.weight,
#                 out_proj_bias=self.c_proj.bias,
#                 use_separate_proj_weight=True,
#                 training=self.training,
#                 need_weights=False
#             )

#         # (5) 최종 출력 정리: (HW+1, B, C) -> (B, C, HW+1)
#         x = x.permute(1, 2, 0)
#         # global_feat: CLS 토큰 위치 (index 0)
#         global_feat = x[:, :, 0]
#         # feature_map: 나머지 토큰을 (B, C, H, W)로 복원
#         feature_map = x[:, :, 1:].reshape(B, -1, H, W)
#         return global_feat, feature_map


# implement attention module for v-v self-attention
class VisionAttention(nn.Module):
    def __init__(self, out_dim, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., settings=''):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.settings = settings

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # original self-attention for the original path
        attn_ori = (q @ k.transpose(-2, -1)) * self.scale
        attn_ori = attn_ori.softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)

        # replace k & q by v
        k = v
        q = k

        # resnets have only one self-attention, norm and larger scale perform better
        if self.settings == 'resnet':
            k = k / (k.norm(p=2, dim=-1, keepdim=True) + 1e-6)
            q = k
            scale = self.scale * 8
        else:
            scale = self.scale

        # self-attention, higher temperate for resnets performs better
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = (attn).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # clip_surgery
        #x = v.transpose(1, 2).reshape(B, N, C) # mask_clip
        x = self.proj_drop(self.proj(x))
        x_ori = self.proj_drop(self.proj(x_ori))
        return [x, x_ori]


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        # self.num_heads = num_heads

        self.attn = None
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_dim = output_dim


    def forward(self, x):

        B, C, H, W = x.shape  # 원래의 공간 크기 추출

        # reform transformer layer after init and load weights, using v only
        if self.attn == None:
            self.attn = VisionAttention(self.output_dim, self.embed_dim, self.num_heads, True)
            self.attn.qkv.weight = torch.nn.Parameter(torch.cat([self.v_proj.weight, self.v_proj.weight, self.v_proj.weight], 0))
            self.attn.qkv.bias = torch.nn.Parameter(torch.cat([self.v_proj.bias, self.v_proj.bias, self.v_proj.bias]))
            self.attn.proj.weight = self.c_proj.weight
            self.attn.proj.bias = self.c_proj.bias

        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        side = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[0] - 1) ** 0.5)

        # update the position embedding during inference for varied input size
        if side != new_side:
            new_pos = self.positional_embedding[1:, :].reshape(-1, side, side, x.shape[-1]).permute(0, 3, 1, 2)
            new_pos = torch.nn.functional.interpolate(new_pos, (new_side, new_side), mode='bilinear')
            new_pos = new_pos.reshape(-1, x.shape[-1], new_side * new_side).transpose(1, 2)
            self.positional_embedding.data = torch.cat([self.positional_embedding[:1, :], new_pos[0]], 0)

        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC


        x, x_ori = self.attn(x.transpose(0, 1))

        # cls token from the original path, and img tokens from the new path
        x[:, 0, :] = x_ori[:, 0, :]

        # original
        # return x

        x = x.permute(0, 2, 1)
        global_feat = x[:, :, 0]
        feature_map = x[:, :, 1:].reshape(B, -1, H, W)

        # x.shape torch.Size([257, 1, 1024])
        # global_feat.shape torch.Size([1, 1024])
        # feature_map.shapex.shape torch.Size([1, 1024, 16, 16])

        return global_feat, feature_map



# @BACKBONES.register_module()
# class CLIPResNet(nn.Module):
#     """
#     A ResNet class that is similar to torchvision's but contains the following changes:
#     - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
#     - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
#     - The final pooling layer is a QKV attention instead of an average pool
#     """

#     def __init__(self, layers, output_dim=512, input_resolution=224, width=64, pretrained=None, **kwargs):
#         super().__init__()
#         self.pretrained = pretrained
#         self.output_dim = output_dim
#         self.input_resolution = input_resolution

#         # the 3-layer stem
#         self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(width // 2)
#         self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(width // 2)
#         self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(width)
#         self.avgpool = nn.AvgPool2d(2)
#         self.relu = nn.ReLU(inplace=True)

#         # residual layers
#         self._inplanes = width  # this is a *mutable* variable used during construction
#         self.layer1 = self._make_layer(width, layers[0])
#         self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
#         self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
#         self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

#     def init_weights(self, pretrained=None):
#         pretrained = pretrained or self.pretrained
#         if isinstance(pretrained, str):
#             checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

#             state_dict = {}

#             for k in checkpoint.keys():
#                 if k.startswith('visual.'):
#                     new_k = k.replace('visual.', '')
#                     state_dict[new_k] = checkpoint[k]

#             u, w = self.load_state_dict(state_dict, False)

#     def _make_layer(self, planes, blocks, stride=1):
#         layers = [Bottleneck(self._inplanes, planes, stride)]

#         self._inplanes = planes * Bottleneck.expansion
#         for _ in range(1, blocks):
#             layers.append(Bottleneck(self._inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x, seg_mask=None):
#         def stem(x):
#             for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
#                 x = self.relu(bn(conv(x)))
#             x = self.avgpool(x)
#             return x


#         x = x.type(self.conv1.weight.dtype)
#         x = stem(x)

#         outs = []
#         x = self.layer1(x)
#         outs.append(x)
#         x = self.layer2(x)
#         outs.append(x)
#         x = self.layer3(x)
#         outs.append(x)
#         x = self.layer4(x)
#         outs.append([x])

#         return tuple(outs)


# @BACKBONES.register_module()
# class CLIPResNetWithAttention(nn.Module):
#     """
#     A ResNet class that is similar to torchvision's but contains the following changes:
#     - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
#     - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
#     - The final pooling layer is a QKV attention instead of an average pool
#     """

#     def __init__(self, layers, output_dim=1024, input_resolution=224, width=64, pretrained=None, **kwargs):
#         super().__init__()
#         self.pretrained = pretrained
#         self.output_dim = output_dim
#         self.input_resolution = input_resolution

#         # the 3-layer stem
#         self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(width // 2)
#         self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(width // 2)
#         self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(width)
#         self.avgpool = nn.AvgPool2d(2)
#         self.relu = nn.ReLU(inplace=True)

#         # residual layers
#         self._inplanes = width  # this is a *mutable* variable used during construction
#         self.layer1 = self._make_layer(width, layers[0])
#         self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
#         self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
#         self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

#         embed_dim = width * 32  # the ResNet feature dimension
#         # original
#         self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, 32, output_dim)
#         # noise_ver
#         # self.attnpool = AttentionPool2d_noise(input_resolution // 32, embed_dim, 32, output_dim)

#     def init_weights(self, pretrained=None):
#         pretrained = pretrained or self.pretrained
#         if isinstance(pretrained, str):
#             checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

#             state_dict = {}

#             for k in checkpoint.keys():
#                 if k.startswith('visual.'):
#                     new_k = k.replace('visual.', '')
#                     state_dict[new_k] = checkpoint[k]

#                     if 'positional_embedding' in new_k:
#                         if self.attnpool.positional_embedding.shape != state_dict[new_k].shape:
#                             cls_pos = state_dict[new_k][0:1, :]
#                             H = W = self.input_resolution // 32
#                             old_h = int(math.sqrt(state_dict[new_k][1:,].shape[0]))
#                             spatial_pos = F.interpolate(state_dict[new_k][1:,].reshape(1, old_h, old_h, cls_pos.shape[1]).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
#                             spatial_pos = spatial_pos.reshape(cls_pos.shape[1], H*W).permute(1, 0)
#                             positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
#                             state_dict[new_k] = positional_embedding
#                             assert self.attnpool.positional_embedding.shape == state_dict[new_k].shape

#             u, w = self.load_state_dict(state_dict, False)

#     def _make_layer(self, planes, blocks, stride=1):
#         layers = [Bottleneck(self._inplanes, planes, stride)]

#         self._inplanes = planes * Bottleneck.expansion
#         for _ in range(1, blocks):
#             layers.append(Bottleneck(self._inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x, seg_mask=None):
#         def stem(x):
#             for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
#                 x = self.relu(bn(conv(x)))
#             x = self.avgpool(x)
#             return x

#         x = x.type(self.conv1.weight.dtype)
#         x = stem(x)

#         outs = []
#         x = self.layer1(x)
#         outs.append(x)
#         x = self.layer2(x)
#         outs.append(x)
#         x = self.layer3(x)
#         outs.append(x)
#         x = self.layer4(x)
#         outs.append(x)

#         assert False
#         # assert False
#         x_global, x_local = self.attnpool(x,seg_mask)
#         outs.append([x_global, x_local])

#         return tuple(outs)




# -----------------------------------------------------------------

import random
import math
from collections import OrderedDict

# 아래 함수들은 노이즈 삽입 및 AdaIN 변환(통계 기반) 연산에 사용됩니다.
def masked_mean_std(x, mask, eps=1e-5):
    """
    x: [B, C, H, W]
    mask: [B, 1, H, W] (float형, 0 또는 1)
    → mask가 1인 영역의 채널별 평균과 표준편차 계산
    """
    B, C, H, W = x.shape
    mask_bc = mask.expand(-1, C, -1, -1)
    masked_x = x * mask_bc
    denom = mask_bc.sum(dim=(2, 3), keepdim=True) + eps
    mean = masked_x.sum(dim=(2, 3), keepdim=True) / denom
    var = ((masked_x - mean) ** 2).sum(dim=(2, 3), keepdim=True) / denom
    std = torch.sqrt(var + eps)
    return mean, std

# def adain(content, style_mean, style_std, eps=1e-5):
#     """
#     content: [B, C, H, W]
#     일반 영역의 스타일 통계 (style_mean, style_std)를 적용하여 OOD 영역의 특징을 정규화
#     AdaIN 공식: F_new = style_std * ((F - μ_F) / σ_F) + style_mean
#     """
#     c_mean = content.mean(dim=(2, 3), keepdim=True)
#     c_std  = content.std(dim=(2, 3), keepdim=True) + eps
#     normalized = (content - c_mean) / c_std
#     return normalized * style_std + style_mean

def adain(content, style_mean, style_std, alpha=0.1, eps=1e-5):
    """
    content: [B, C, H, W]
    style_mean, style_std: 일반 영역의 스타일 통계
    alpha: 스타일 적용 정도 (0이면 원본 유지, 1이면 완전히 적용)
    """
    c_mean = content.mean(dim=(2, 3), keepdim=True)
    c_std  = content.std(dim=(2, 3), keepdim=True) + eps
    normalized = (content - c_mean) / c_std
    stylized = normalized * style_std + style_mean

    # alpha 값을 적용하여 원본 content와 혼합

    if random.random() < 0.5:
        alpha = 0.2
    else:
        alpha = 0.0

    return alpha * stylized + (1 - alpha) * content

# -----------------------------------------------------------------
@BACKBONES.register_module()
class CLIPResNetWithAttention(nn.Module):
    """
    기존 CLIPResNetWithAttention 구조를 유지하면서,
    중간 residual layer(각 layer의 출력)에서 seg_mask가 주어질 경우,
    확률적으로 아래의 두 작업을 수행합니다.
      1. 일반 cls 영역에 대해 클래스 내부 통계 기반 노이즈 삽입
      2. OOD cls 영역은 일반 cls 영역의 통계(style)를 적용하여 AdaIN 식 정규화 수행
    """
    def __init__(self, layers, output_dim=1024, input_resolution=224, width=64, pretrained=None, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # --- 추가 파라미터: 노이즈 및 OOD 정렬 관련 ---
        self.ood_label = 19            # OOD 클래스 라벨 (예: 255)
        self.noise_and_adain_prob = 0.5   # 확률적으로 적용 (예: 50%)
        self.noise_scale = 0.0001          # 노이즈 강도

        # stem (변경 없음)
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu    = nn.ReLU(inplace=True)

        # residual layers (기존 Bottleneck, BN 기반)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # ResNet feature 차원
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, 32, output_dim)

    def init_weights(self, pretrained=None):
        # 기존 pretrain weight 로딩 로직 (변경 없음)
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()
            state_dict = {}
            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]
                    if 'positional_embedding' in new_k:
                        if self.attnpool.positional_embedding.shape != state_dict[new_k].shape:
                            warnings.warn(
                                f'Resize the pos_embed shape from {state_dict[new_k].shape} '
                                f'to {self.attnpool.positional_embedding.shape}',
                                stacklevel=2)
                            cls_pos = state_dict[new_k][0:1, :]
                            H = W = self.input_resolution // 32
                            old_h = int(math.sqrt(state_dict[new_k][1:].shape[0]))
                            spatial_pos = F.interpolate(state_dict[new_k][1:].reshape(1, old_h, old_h, cls_pos.shape[1]).permute(0, 3, 1, 2),
                                                        size=(H, W), mode='bilinear')
                            spatial_pos = spatial_pos.reshape(cls_pos.shape[1], H * W).permute(1, 0)
                            positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                            state_dict[new_k] = positional_embedding
                            assert self.attnpool.positional_embedding.shape == state_dict[new_k].shape
            u, w = self.load_state_dict(state_dict, strict=False)
            warnings.warn(f'{u}, {w} are misaligned params in CLIPResNet', stacklevel=2)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    ### 추가: 중간 residual layer에서 노이즈 삽입 및 OOD 영역 스타일 정렬 적용 함수
    def _apply_noise_and_style(self, x, seg_mask):
        """
        x: [B, C, H, W] feature map
        seg_mask: [B, 1, H_orig, W_orig] segmentation mask
        수행 과정:
         1. seg_mask를 feature map 크기로 다운샘플링
         2. OOD 영역 (seg_mask == self.ood_label)와 일반 영역(normal)을 분리
         3. 일반 영역의 통계(채널별 mean, std)를 계산하고, 해당 영역에 대해 노이즈 삽입
         4. OOD 영역은 AdaIN 공식에 따라 일반 영역의 통계(style)를 적용하여 변환
         5. 두 영역을 재결합하여 반환
        """
        # seg_mask 다운샘플링
        seg_mask_down = F.interpolate(seg_mask.float(), size=x.shape[-2:], mode='nearest')
        ood_mask = (seg_mask_down == self.ood_label).float()   # OOD 영역: 1, 나머지: 0
        normal_mask = 1.0 - ood_mask                           # 일반 영역

        # (1) 일반 영역 통계 계산 (클래스 내부 통계 기반)
        n_mean, n_std = masked_mean_std(x, normal_mask)
        # (2) 일반 영역에 대해 노이즈 삽입
        noise = torch.randn_like(x) * (n_std * self.noise_scale)
        x_noisy = x + noise * normal_mask.expand_as(x)
        # (3) OOD 영역 특징은 일반 영역 스타일로 정렬 (AdaIN 식 적용, 학습 파라미터 없음)
        x_transformed = adain(x_noisy, n_mean, n_std)
        # (4) 두 영역 혼합: 일반 영역은 x_noisy, OOD 영역은 x_transformed 사용
        return normal_mask.expand_as(x) * x_noisy + ood_mask.expand_as(x) * x_transformed

    def forward(self, x, seg_mask=None):
        """
        seg_mask가 None이면 inference 모드로, 노이즈/스타일 변환 없이 처리합니다.
        seg_mask가 주어지고 training 중이며 확률 조건을 만족하면,
        각 residual layer 이후에 _apply_noise_and_style() 함수를 호출하여
        일반 영역에 노이즈 삽입 및 OOD 영역에 대한 스타일 정렬을 적용합니다.
        """
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)

        outs = []

        # Layer 1
        x = self.layer1(x)
        if (seg_mask is not None) and self.training and (random.random() < self.noise_and_adain_prob):
            ## 수정: Layer 1 후 노이즈 및 스타일 변환 적용
            x = self._apply_noise_and_style(x, seg_mask)
        outs.append(x)

        # Layer 2
        x = self.layer2(x)
        # if (seg_mask is not None) and self.training and (random.random() < self.noise_and_adain_prob):
            ### 수정: Layer 2 후 노이즈 및 스타일 변환 적용
            # x = self._apply_noise_and_style(x, seg_mask)
        outs.append(x)

        # Layer 3
        x = self.layer3(x)
        # if (seg_mask is not None) and self.training and (random.random() < self.noise_and_adain_prob):
        #     ### 수정: Layer 3 후 노이즈 및 스타일 변환 적용
        #     # x = self._apply_noise_and_style(x, seg_mask)
        outs.append(x)

        # Layer 4
        x = self.layer4(x)
        # if (seg_mask is not None) and self.training and (random.random() < self.noise_and_adain_prob):
        #     ### 수정: Layer 4 후 노이즈 및 스타일 변환 적용
        #     # x = self._apply_noise_and_style(x, seg_mask)
        outs.append(x)

        x_global, x_local = self.attnpool(x)
        outs.append([x_global, x_local])

        return tuple(outs)
