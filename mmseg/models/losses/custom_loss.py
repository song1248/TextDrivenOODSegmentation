import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import LOSSES
from mmseg.models.losses.utils import weight_reduce_loss


# @LOSSES.register_module(force=True)
# class MaskContrastiveLoss(nn.Module):
#     """단일 채널(또는 포인트 기준) 마스크 예측에 대한 Contrastive Loss 예시.

#     여기서는 mask_point_targets=1이면 OOD, 0이면 ID로 해석.
#     따라서 pred_prob는 "이 픽셀이 OOD일 확률"이 되도록 sigmoid()를 적용.

#     Args:
#         margin (float): ID vs OOD 사이의 마진 (옵션).
#         reduction (str): 'none' | 'mean' | 'sum'.
#         loss_weight (float): 최종 로스에 곱할 가중치.
#         loss_name (str): 로스 이름.
#     """

#     def __init__(self,
#                  margin=1.0,
#                  reduction='mean',
#                  loss_weight=1.0,
#                  loss_name='loss_mask_contrastive'):
#         super(MaskContrastiveLoss, self).__init__()
#         self.margin = margin
#         self.reduction = reduction
#         self.loss_weight = loss_weight
#         self._loss_name = loss_name

#     def forward(self,
#                 mask_point_preds,
#                 mask_point_targets,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None,
#                 **kwargs):
#         """
#         Args:
#             mask_point_preds (Tensor): (N, P) 형태의 마스크 로짓 예측.
#                                        여기서는 "OOD 확률"을 예측한다고 간주.
#             mask_point_targets (Tensor): (N, P) 형태의 GT (1=OOD, 0=ID).
#             weight (Tensor, optional): (N, P) 동일 shape 등으로 주어질 수 있는 가중치.
#             avg_factor (float, optional): mean 시 나누어줄 값.
#             reduction_override (str, optional): 'none'|'mean'|'sum'으로 override 가능.

#         Returns:
#             Tensor: shape=()의 스칼라 로스.
#         """
#         reduction = reduction_override if reduction_override else self.reduction

#         # 1) sigmoid로 OOD 확률로 변환
#         pred_prob = torch.sigmoid(mask_point_preds)  # (N, P), 0~1

#         # 2) ood_mask = mask_point_targets (1=OOD, 0=ID)
#         ood_mask = mask_point_targets  # (N, P)

#         # 3) ID vs OOD가 구분되도록 대조(contrast) 로직 구성
#         #    - ID(ood=0)는 pred_prob≈0이 되어야 손실↓
#         #    - OOD(ood=1)는 pred_prob≈1이 되어야 손실↓

#         # 간단히:
#         #    l_N = pred_prob (OOD일 확률)
#         #    ID 픽셀 => l_cl = max(0, margin - l_N)
#         #    OOD 픽셀 => l_cl = l_N
#         #    => ID 픽셀이면 pred_prob를 0쪽으로,
#         #       OOD 픽셀이면 pred_prob를 1쪽으로 유도
#         l_N = pred_prob  # (N, P)

#         l_cl_id = torch.clamp(self.margin - l_N, min=0.0)  # ID일 때
#         l_cl_ood = l_N                                     # OOD일 때

#         l_cl = (1 - ood_mask) * l_cl_id + ood_mask * l_cl_ood

#         # 4) 제곱항(0.5*(l_cl^2))을 적용해 스무싱
#         loss = 0.5 * (l_cl ** 2)

#         # 5) weight & reduction
#         loss = weight_reduce_loss(
#             loss,
#             weight=weight,
#             reduction=reduction,
#             avg_factor=avg_factor)

#         return self.loss_weight * loss

#     @property
#     def loss_name(self):
#         return self._loss_name

@LOSSES.register_module(force=True)
class MaskContrastiveLoss(nn.Module):
    """마스크 기반 Contrastive Loss. (mask_point_targets=1 => OOD, 0 => ID)"""

    def __init__(self,
                 margin=1.0,
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='loss_mask_contrastive'):
        super(MaskContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                mask_point_preds,
                mask_point_targets,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        # shape: (N, P)
        reduction = reduction_override if reduction_override else self.reduction

        # 1) sigmoid
        pred_prob = torch.sigmoid(mask_point_preds)  # (N, P)

        # 2) mask_point_targets -> 1=OOD, 0=ID
        ood_mask = mask_point_targets  # (N, P)

        # 3) l_cl
        l_N = pred_prob  # (N, P)
        l_cl_id = torch.clamp(self.margin - l_N, min=0.0)  # ID
        l_cl_ood = l_N                                     # OOD
        
        l_cl = (1 - ood_mask) * l_cl_id + ood_mask * l_cl_ood

        # 4) 0.5*(l_cl^2)
        loss = 0.5 * (l_cl**2)

        # 5) reduction
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self._loss_name
