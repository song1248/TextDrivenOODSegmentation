import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from mmseg.ops import resize
from mmseg.core import add_prefix
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor

from ..backbones.clip import *
from ..backbones.utils import tokenize

@SEGMENTORS.register_module()
class TextDrivenOODSeg(BaseSegmentor):

    def __init__(self,
                 backbone,
                 text_encoder,
                 decode_head,
                 class_names,
                 context_length,
                 context_decoder=None,
                 token_embed_dim=512,
                 text_dim=512,
                 neck=None,
                 identity_head=None,
                 visual_reg=True,
                 textual_reg=True,
                 pretrained=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **args):

        super(TextDrivenOODSeg, self).__init__(init_cfg)

        self.tau = 0.07
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.num_classes = len(class_names)
        self.context_length = context_length
        self.visual_reg = visual_reg
        self.textual_reg = textual_reg


        self.learnable_cls_num = 50 # 75 # learnable class 개수
        decode_head['num_queries'] += self.learnable_cls_num
        decode_head['pixel_decoder']['num_text_embeds'] += self.learnable_cls_num
        # decode_head['loss_cls']['class_weight'] = [1.0] * (20+self.learnable_cls_num) + [0.1]
        self.score_map_method = 'max' # 'mean', 'max', 'matching'


        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'
            if 'CLIP-ViT-B-16.pt' in backbone.get('pretrained'):
                text_encoder.pretrained = 'pretrained/CLIP-ViT-B-16.pt'
            elif 'RN50.pt' in backbone.get('pretrained'):
                text_encoder.pretrained = 'pretrained/RN50.pt'
            else:
                raise AssertionError("wrong pretrained model. please modify config or code")

        # build components
        self.backbone = builder.build_backbone(backbone); self.backbone.init_weights()
        self.text_encoder = builder.build_backbone(text_encoder); self.text_encoder.init_weights()
        self.neck = builder.build_neck(neck) if neck is not None else None
        self.context_decoder = builder.build_backbone(context_decoder) if context_decoder is not None else None

        # 🔹 1. 먼저 모든 가중치를 Freeze (백본 전체 동결)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # requires_grad False
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False

        # build head
        self.decode_head = builder.build_head(decode_head) if decode_head is not None else None
        self.identity_head = builder.build_head(identity_head) if identity_head is not None else None

        # coop
        self.text_encoder.to('cuda')
        prompt_num = self.text_encoder.context_length - self.context_length
        self.texts = torch.cat([tokenize(c, context_length=context_length) for c in class_names]).to('cuda')
        self.contexts = nn.Parameter(torch.randn(1, prompt_num, token_embed_dim))
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)

        nn.init.trunc_normal_(self.contexts)
        nn.init.trunc_normal_(self.gamma)

        # vision regularization
        self.reg_E0 = copy.deepcopy(self.backbone)
        self.reg_E0.eval()

        # language regularization
        self.reg_T0 = torch.cat([tokenize(
                texts=f"a clean origami of a {c}",
                context_length=self.text_encoder.context_length)
            for c in class_names]).to('cuda')

        # 학습 할 prompt 생성

        # self.texts 에 learnable prompt 를 추가해야함

        if self.learnable_cls_num > 0:
            self.negative_learnable_prompt = nn.Parameter(torch.randn(self.learnable_cls_num, prompt_num, token_embed_dim))
        else:
            self.negative_learnable_prompt = None



        # regularization text
        if self.learnable_cls_num > 0:

            # 파일 읽어서 리스트로 변환
            file_path = "/home/jovyan/work/NegLabel/selected_neg_labels_ver2.txt"
            negative_words = []
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    # 공백 문자 제거
                    line = line.strip()
                    # "This is a "와 "photo" 사이의 텍스트 추출
                    if "This is a " in line and "photo" in line:
                        start = line.find("This is a ") + len("This is a ")
                        end = line.find("photo")
                        negative_word = line[start:end].strip()
                        negative_words.append(negative_word)

            self.neg_reg_T0 = torch.cat([tokenize(
                    texts=f"a clean origami of a {c}",
                    context_length=self.text_encoder.context_length)
                for c in negative_words]).to('cuda')

            self.neg_reg_T0 = F.normalize(self.text_encoder(self.neg_reg_T0, context=None), dim=-1, p=2)

            # 텐서를 M 등분으로 나눌 수 있는 크기 계산
            M = self.learnable_cls_num
            num_rows = self.neg_reg_T0.shape[0]
            split_size = num_rows // M  # 각 등분의 행 개수

            # 텐서를 M 등분으로 나누기 (split_size로 자르고 나머지는 버림)
            splits = torch.split(self.neg_reg_T0[:split_size * M], split_size)

            # 각 등분끼리 평균 계산
            self.M_neg_reg_T0 = torch.stack([split.mean(dim=0) for split in splits])
            self.reg_T0 = F.normalize(self.text_encoder(self.reg_T0, context=None), dim=-1, p=2)

            self.concat_T0 = torch.cat([self.reg_T0, self.M_neg_reg_T0], dim=0)  # dim=0: 행 기준으로 concat
        else:
            self.reg_T0 = F.normalize(self.text_encoder(self.reg_T0, context=None), dim=-1, p=2)
            self.concat_T0 = self.reg_T0




        # origin
        # self.reg_I = torch.arange(self.num_classes, device='cuda')
        self.reg_I = torch.arange(self.num_classes + self.learnable_cls_num, device='cuda')



    def extract_feat(self, img, seg_mask=None):
        x = self.backbone(img, seg_mask)
        return x

    def after_extract_feat(self, x):
        x_orig = list(x[:-1])
        global_feat, visual_embeddings = x[-1]

        using_resnet = False
        if len(global_feat.shape) == 2:
            using_resnet = True

        if using_resnet:
            global_feat = global_feat.unsqueeze(dim=1)


        b_size = global_feat.shape[0]


        visual_context = torch.cat([global_feat, visual_embeddings.flatten(-2).permute(0, 2, 1)], dim=1)
        # text_embeddings = self.text_encoder(self.texts, context=self.contexts).expand(b_size, -1, -1)

        text_embeddings = self.text_encoder(self.texts, context=self.contexts, learnable_text=self.negative_learnable_prompt).expand(b_size, -1, -1)

        if self.context_decoder is not None:
            text_diff = self.context_decoder(text_embeddings, visual_context)
            text_embeddings = text_embeddings + self.gamma * text_diff
        ret_text_emb = text_embeddings

        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text_embeddings)


        if using_resnet:
            x_orig[-1] = torch.cat([x_orig[-1], score_map], dim=1)

        # assert False
        return x_orig, score_map, ret_text_emb, global_feat

    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):


        x = self.extract_feat(img, gt_semantic_seg) # seg_mask가 들어감 -> Noise 줌
        x_orig, score_map, text_emb, global_feat = self.after_extract_feat(x)
        x = list(self.neck(x_orig)) if self.neck is not None else x_orig



        losses = dict()

        # language regularization
        if self.textual_reg is True:

            ####### original start
            # content_score = torch.einsum('blc,kc->bkl', F.normalize(text_emb, dim=-1, p=2), self.reg_T0.detach())
            content_score = torch.einsum('blc,kc->bkl', F.normalize(text_emb, dim=-1, p=2), self.concat_T0.detach())
            loss_reg_l = F.cross_entropy(content_score,
                self.reg_I.expand(content_score.shape[0], -1),
                reduction='mean')
            ####### original end

            loss_reg_l = {'loss' : loss_reg_l}
            losses.update(add_prefix(loss_reg_l, 'reg.textual'))

        # vision regularization
        if self.visual_reg is True:
            with torch.no_grad():
                global_feat_0, _ = self.reg_E0(img)[-1]

            if global_feat.shape != global_feat_0.shape:
                global_feat_0 = global_feat_0.unsqueeze(dim=1)

            loss_reg_v = nn.MSELoss(reduction='mean')(global_feat, global_feat_0) * 10

            # Cosine Similarity Loss 추가
            cosine_similarity = F.cosine_similarity(global_feat, global_feat_0, dim=-1)
            loss_reg_v_cos = (1 - cosine_similarity).mean() * 2.0  # 유사도를 높이기 위해 2.0 배 적용

            # 최종 Vision Regularization Loss
            loss_reg_v_total = loss_reg_v + loss_reg_v_cos
            loss_reg_v = {'loss' : loss_reg_v_total}
            losses.update(add_prefix(loss_reg_v, 'reg.visual'))


        # score_map의 max 값을 찾기
        score_map_max_per_class, _ = torch.max(score_map.view(score_map.shape[0], score_map.shape[1], -1), dim=-1)
        # assert False


        # $
        if self.score_map_method == 'mean':  # 방법 1: 평균
            score_map = torch.cat([
                score_map[:, :-(self.learnable_cls_num+1), :, :],
                torch.mean(score_map[:, -(self.learnable_cls_num+1):, :, :], dim=1, keepdim=True)[0]
            ], dim=1)
        elif self.score_map_method == 'matching':  # 방법 3: bipartite matching (gt_semantic_seg의 19번째 cls 이용)
            # gt_semantic_seg에서 19번째 클래스 영역에 대한 바이너리 마스크 생성
            gt_mask = (gt_semantic_seg == 19).float()  # shape: [B, 1, H, W]

            # learnable 채널들 (마지막 (learnable_cls_num+1) 채널; 기존 코드 slicing 유지)
            learnable_channels = score_map[:, -(self.learnable_cls_num+1):, :, :]  # shape: [B, M+1, H, W]

            # gt_mask의 해상도를 learnable_channels에 맞게 조정
            gt_mask = F.interpolate(gt_mask, size=learnable_channels.shape[-2:], mode='nearest')

            # 각 채널에 대해 gt_mask 영역의 평균 값을 matching score로 계산 (채널별 활성화 정도)
            matching_scores = torch.sum(learnable_channels * gt_mask, dim=(2, 3)) / (torch.sum(gt_mask, dim=(2, 3)) + 1e-6)  # shape: [B, M+1]


            # 각 배치별로 가장 높은 matching score를 가지는 채널의 인덱스 선택
            best_channel_idx = matching_scores.argmax(dim=1)  # shape: [B]

            # 각 배치마다 선택된 채널의 feature map을 추출하고 4D 텐서 [B, 1, H, W]로 만듦
            best_channel = [learnable_channels[i, best_channel_idx[i], :, :].unsqueeze(0) for i in range(score_map.shape[0])]
            best_channel = torch.stack(best_channel, dim=0)  # shape: [B, 1, H, W]

            # 기존 채널과 선택된 채널을 합침
            score_map = torch.cat([
                score_map[:, :-(self.learnable_cls_num+1), :, :],
                best_channel
            ], dim=1)
        else:
            # fallback: 기본은 평균 방식 사용
            score_map = torch.cat([
                score_map[:, :-(self.learnable_cls_num+1), :, :],
                torch.max(score_map[:, -(self.learnable_cls_num+1):, :, :], dim=1, keepdim=True)[0]
            ], dim=1)
        # $

        # vision-language regularization
        if self.identity_head is not None:
            loss_score_map = self.identity_head.forward_train(
                score_map/self.tau, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_score_map, 'scr_map'))

        # decode head loss
        # assert False

        # original  start
        # loss_decode = self.decode_head.forward_train(
        #     x, text_emb, img_metas, gt_semantic_seg, self.train_cfg, kwargs['gt_labels'], kwargs['gt_masks'])
        # original  end


        loss_decode = self.decode_head.forward_train(
            x, text_emb, img_metas, gt_semantic_seg, self.train_cfg, kwargs['gt_labels'], kwargs['gt_masks'])

        losses.update(add_prefix(loss_decode, 'decode'))


        return losses

    def encode_decode(self, img, img_metas):
        x = self.extract_feat(img) # seg_mask가 안들어감 -> Noise안줌
        x_orig, score_map, text_emb, global_feat = self.after_extract_feat(x)
        x = list(self.neck(x_orig)) if self.neck is not None else x_orig

        out = self.decode_head.forward_test(
            x, text_emb, img_metas, self.test_cfg)
        out = resize(
            input=out,
            size=img.shape[-2:],
            mode='bilinear',
            align_corners=False)

        return out

    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""


        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=False)

        if torch.isnan(seg_logit).any():
            raise ValueError('NaN detected in segmentation logits')

        return seg_logit

    def inference(self, img, img_meta, rescale):


        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        self.eval()
        with torch.no_grad():

            assert self.test_cfg.mode in ['slide', 'whole']
            ori_shape = img_meta[0]['ori_shape']
            # ori_shape = img_meta.data[0]['ori_shape']
            assert all(_['ori_shape'] == ori_shape for _ in img_meta)
            if self.test_cfg.mode == 'slide':
                seg_logit = self.slide_inference(img, img_meta, rescale)
            else:
                seg_logit = self.whole_inference(img, img_meta, rescale)


            # import pickle

            # with open('seg_logit.pkl', 'wb') as f:
            #     pickle.dump(seg_logit, f)

            # assert False

            # assert False

            output = F.softmax(seg_logit, dim=1)


            flip = img_meta[0]['flip']
            if flip:
                flip_direction = img_meta[0]['flip_direction']
                assert flip_direction in ['horizontal', 'vertical']
                if flip_direction == 'horizontal':
                    output = output.flip(dims=(3, ))
                elif flip_direction == 'vertical':
                    output = output.flip(dims=(2, ))

            return output

    # # origin
    # def simple_test(self, img, img_meta, rescale=True):
    #     """Simple test with single image."""
    #     seg_logit = self.inference(img, img_meta, rescale)
    #     seg_pred = seg_logit.argmax(dim=1)
    #     if torch.onnx.is_in_onnx_export():
    #         # our inference backend only support 4D output
    #         seg_pred = seg_pred.unsqueeze(0)
    #         return seg_pred
    #     seg_pred = seg_pred.cpu().numpy()
    #     # unravel batch dim
    #     seg_pred = list(seg_pred)
    #     return seg_pred
    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if self.save_seg_logit is True:
            self.seg_logit = seg_logit.cpu().numpy()
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        seg_logit = seg_logit.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        seg_logit = list(seg_logit)

        return seg_pred, seg_logit

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred
