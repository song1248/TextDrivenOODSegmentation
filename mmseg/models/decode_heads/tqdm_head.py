# import copy
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.cnn import Conv2d, build_plugin_layer, caffe2_xavier_init, ConvModule
# from mmcv.cnn.bricks.transformer import (build_positional_encoding,
#                                          build_transformer_layer_sequence)
# from mmcv.ops import point_sample
# from mmcv.runner import ModuleList, force_fp32
# from mmseg.models.builder import HEADS, build_loss
# from mmseg.models.decode_heads.decode_head import BaseDecodeHead

# from ...core import build_sampler, multi_apply, reduce_mean
# from ..builder import build_assigner
# from ..utils import get_uncertain_point_coords_with_randomness


# @HEADS.register_module()
# class tqdmHead(BaseDecodeHead):
#     """
#     This head implements a Mask2Former-style decoder head in which each query is assigned a fixed class.
#     In this design, the first `num_classes` queries correspond to fixed semantic classes and the remaining
#     queries (M = num_queries - num_classes) are learnable for more flexible mask prediction.

#     Args:
#         in_channels (int): Number of input channels from the backbone.
#         feat_channels (int): Number of channels for intermediate features.
#         out_channels (int): Number of output channels for mask prediction.
#         num_things_classes (int): Number of 'thing' classes.
#         num_stuff_classes (int): Number of 'stuff' classes.
#         num_queries (int): Total number of queries. The first num_classes queries are fixed.
#         num_transformer_feat_level (int): Number of multi-scale feature levels for the transformer.
#         pixel_decoder (dict): Configuration for the pixel decoder.
#         enforce_decoder_input_project (bool): Whether to enforce input projection.
#         transformer_decoder (dict): Configuration for the transformer decoder.
#         positional_encoding (dict): Configuration for the positional encoding.
#         loss_cls (dict): Configuration for the classification loss.
#         loss_mask (dict): Configuration for the mask loss.
#         loss_dice (dict): Configuration for the dice loss.
#         train_cfg (dict): Training configuration.
#         test_cfg (dict): Testing configuration.
#         init_cfg (dict): Initialization configuration.
#         text_proj (dict): Configuration for projecting text embeddings.
#     """
#     def __init__(self,
#                  in_channels,
#                  feat_channels,
#                  out_channels,
#                  num_things_classes=80,
#                  num_stuff_classes=53,
#                  num_queries=100,
#                  num_transformer_feat_level=3,
#                  pixel_decoder=None,
#                  enforce_decoder_input_project=False,
#                  transformer_decoder=None,
#                  positional_encoding=None,
#                  loss_cls=None,
#                  loss_mask=None,
#                  loss_dice=None,
#                  train_cfg=None,
#                  test_cfg=None,
#                  init_cfg=None,
#                  text_proj=None,
#                  **kwargs):
#         # Initialize the base decode head with multiple feature inputs.
#         super(tqdmHead, self).__init__(
#             in_channels=in_channels,
#             channels=feat_channels,
#             num_classes=(num_things_classes + num_stuff_classes),
#             init_cfg=init_cfg,
#             input_transform='multiple_select',
#             **kwargs)

#         # Total number of classes and queries.
#         self.num_things_classes = num_things_classes
#         self.num_stuff_classes = num_stuff_classes
#         self.num_classes = self.num_things_classes + self.num_stuff_classes  # e.g., 80 + 53 = 133
#         self.num_queries = num_queries  # Total queries; first num_classes queries are fixed.
#         self.num_transformer_feat_level = num_transformer_feat_level

#         # Extract number of heads and decoder layers from the transformer_decoder configuration.
#         self.num_heads = transformer_decoder.transformerlayers.attn_cfgs.num_heads
#         self.num_transformer_decoder_layers = transformer_decoder.num_layers

#         # Ensure that the pixel_decoder uses the same number of transformer feature levels.
#         assert pixel_decoder.encoder.transformerlayers.attn_cfgs[0].num_levels == num_transformer_feat_level

#         # Deep copy the pixel_decoder configuration and update with current parameters.
#         pixel_decoder_ = copy.deepcopy(pixel_decoder)
#         pixel_decoder_.update(
#             in_channels=in_channels,
#             feat_channels=feat_channels,
#             out_channels=out_channels,
#             text_proj=text_proj)
#         # Build the pixel decoder module.
#         self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]

#         # Build the transformer decoder module.
#         self.transformer_decoder = build_transformer_layer_sequence(transformer_decoder)
#         self.decoder_embed_dims = self.transformer_decoder.embed_dims

#         # Create input projection layers for each feature level.
#         self.decoder_input_projs = ModuleList()
#         for _ in range(num_transformer_feat_level):
#             # If the embedding dimensions differ or projection is enforced, use a 1x1 conv.
#             if (self.decoder_embed_dims != feat_channels or enforce_decoder_input_project):
#                 self.decoder_input_projs.append(
#                     Conv2d(feat_channels, self.decoder_embed_dims, kernel_size=1))
#             else:
#                 self.decoder_input_projs.append(nn.Identity())

#         # Build the positional encoding module.
#         self.decoder_positional_encoding = build_positional_encoding(positional_encoding)

#         # Query embeddings for initial query features.
#         # Shape: (num_queries, feat_channels)
#         self.query_embed = nn.Embedding(self.num_queries, feat_channels)

#         # Text projection: project input text features to the desired dimension.
#         self.text_proj = nn.Sequential(
#             nn.Linear(text_proj.text_in_dim, text_proj.text_out_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(text_proj.text_out_dim, text_proj.text_out_dim))

#         # Level embedding for each transformer feature level.
#         # Shape: (num_transformer_feat_level, feat_channels)
#         self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)

#         # Classification head: maps query features to class logits.
#         # Output dimension: num_classes + 1 (background included).
#         self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)

#         # Mask head: maps query features to mask embeddings.
#         # Final output dimension is out_channels.
#         self.mask_embed = nn.Sequential(
#             nn.Linear(feat_channels, feat_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(feat_channels, feat_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(feat_channels, out_channels))

#         # conv_seg is not used (set to None to fix a bug).
#         self.conv_seg = None

#         self.test_cfg = test_cfg
#         self.train_cfg = train_cfg

#         # Build assigner and sampler for training if train_cfg is provided.
#         if train_cfg:
#             self.assigner = build_assigner(self.train_cfg.assigner)
#             self.sampler = build_sampler(self.train_cfg.sampler, context=self)
#             self.num_points = self.train_cfg.get('num_points', 12544)
#             self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
#             self.importance_sample_ratio = self.train_cfg.get('importance_sample_ratio', 0.75)

#         # Build loss functions.
#         self.class_weight = loss_cls.class_weight
#         self.loss_cls = build_loss(loss_cls)
#         self.loss_mask = build_loss(loss_mask)
#         self.loss_dice = build_loss(loss_dice)

#         # Set the mask prediction method.
#         # Options include: 'max', 'mean', 'matching', 'ood_scoring', 'ood_scoring_contrastive'
#         self.mask_pred_method = 'max'
#         # M = number of learnable queries = num_queries - num_classes
#         self.M = self.num_queries - self.num_classes

#     def init_weights(self):
#         """Initialize weights for the decoder input projections, pixel decoder, and transformer decoder."""
#         for m in self.decoder_input_projs:
#             if isinstance(m, Conv2d):
#                 caffe2_xavier_init(m, bias=0)

#         self.pixel_decoder.init_weights()

#         for p in self.transformer_decoder.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_normal_(p)

#     def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
#                     gt_masks_list, img_metas):
#         """
#         Compute classification and mask targets for all images for a given decoder layer.

#         Args:
#             cls_scores_list (list[Tensor]): List of classification logits per image.
#                 Each tensor shape: (num_queries, cls_out_channels)
#             mask_preds_list (list[Tensor]): List of mask predictions per image.
#                 Each tensor shape: (num_queries, H, W)
#             gt_labels_list (list[Tensor]): List of ground truth labels per image.
#                 Each tensor shape: (num_gts,)
#             gt_masks_list (list[Tensor]): List of ground truth masks per image.
#                 Each tensor shape: (num_gts, H, W)
#             img_metas (list[dict]): List of image metadata.

#         Returns:
#             tuple: (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
#                     num_total_pos, num_total_neg)
#             - labels_list: List of target labels per image. Each tensor shape: (num_classes,)
#               (Fixed queries per class are used.)
#             - label_weights_list: List of weights for classification loss. Each tensor shape: (num_classes,)
#             - mask_targets_list: List of target masks for positive samples.
#             - mask_weights_list: List of weights for mask loss. Each tensor shape: (num_classes,)
#             - num_total_pos: Total number of positive samples.
#             - num_total_neg: Total number of negative samples.
#         """
#         (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
#          pos_inds_list, neg_inds_list) = multi_apply(
#             self._get_target_single, cls_scores_list,
#             mask_preds_list, gt_labels_list,
#             gt_masks_list, img_metas)

#         num_total_pos = sum((inds.numel() for inds in pos_inds_list))
#         num_total_neg = sum((inds.numel() for inds in neg_inds_list))
#         return (labels_list, label_weights_list, mask_targets_list,
#                 mask_weights_list, num_total_pos, num_total_neg)

#     def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks, img_metas):
#         """
#         Compute targets for a single image.

#         Args:
#             cls_score (Tensor): Classification logits for one image.
#                 Shape: (num_queries, cls_out_channels)
#             mask_pred (Tensor): Mask predictions for one image.
#                 Shape: (num_queries, H, W)
#             gt_labels (Tensor): Ground truth labels.
#                 Shape: (num_gts,)
#             gt_masks (Tensor): Ground truth masks.
#                 Shape: (num_gts, H, W)
#             img_metas (dict): Image metadata.

#         Returns:
#             tuple: (labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds)
#             - labels: Target labels for queries. Shape: (num_classes,)
#             - label_weights: Weights for the classification loss. Shape: (num_classes,)
#             - mask_targets: Target masks for positive samples.
#             - mask_weights: Weights for the mask loss. Shape: (num_classes,)
#             - pos_inds: Indices of positive samples.
#             - neg_inds: Indices of negative samples.
#         """
#         # Determine number of queries and ground truth objects.
#         num_queries = cls_score.shape[0]
#         num_gts = gt_labels.shape[0]

#         # Sample a set of points uniformly within the masks.
#         # point_coords: (1, num_points, 2) with values in [0,1]
#         point_coords = torch.rand((1, self.num_points, 2), device=cls_score.device)
#         # Sample mask predictions at these points.
#         # Resulting shape: (num_queries, num_points)
#         mask_points_pred = point_sample(mask_pred.unsqueeze(1),
#                                         point_coords.repeat(num_queries, 1, 1)).squeeze(1)
#         # Sample ground truth masks at these points.
#         # Resulting shape: (num_gts, num_points)
#         gt_points_masks = point_sample(gt_masks.unsqueeze(1).float(),
#                                        point_coords.repeat(num_gts, 1, 1)).squeeze(1)

#         # Assign predictions to ground truth.
#         assign_result = self.assigner.assign(cls_score, mask_points_pred,
#                                              gt_labels, gt_points_masks, img_metas)
#         sampling_result = self.sampler.sample(assign_result, mask_pred, gt_masks)
#         pos_inds = sampling_result.pos_inds
#         neg_inds = sampling_result.neg_inds

#         # Create target labels for fixed queries (using num_classes instead of num_queries).
#         labels = gt_labels.new_full((self.num_classes,), self.num_classes, dtype=torch.long)
#         # Set labels for positive indices based on assigned ground truth.
#         labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]

#         # Set label weights (all ones).
#         label_weights = gt_labels.new_ones((self.num_classes,))

#         # For mask targets, only the positive samples are considered.
#         mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
#         mask_weights = mask_pred.new_zeros((self.num_classes,))
#         mask_weights[pos_inds] = 1.0

#         return (labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds)

#     def loss_single(self, cls_scores, mask_preds, gt_labels_list,
#                     gt_masks_list, img_metas):
#         """
#         Compute losses for outputs from a single decoder layer.

#         Args:
#             cls_scores (Tensor): Classification logits for all images.
#                 Shape: (batch_size, num_queries, cls_out_channels)
#             mask_preds (Tensor): Mask predictions for all images.
#                 Shape: (batch_size, num_queries, H, W)
#             gt_labels_list (list[Tensor]): Ground truth labels for each image.
#                 Each tensor shape: (num_gts,)
#             gt_masks_list (list[Tensor]): Ground truth masks for each image.
#                 Each tensor shape: (num_gts, H, W)
#             img_metas (list[dict]): Image metadata.

#         Returns:
#             tuple: (loss_cls, loss_mask, loss_dice)
#         """
#         num_imgs = cls_scores.size(0)
#         # Split predictions per image.
#         cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
#         mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

#         # Get targets for all images.
#         (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
#          num_total_pos, num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list,
#                                                           gt_labels_list, gt_masks_list, img_metas)
#         # Stack targets.
#         # labels: (batch_size, num_classes)
#         labels = torch.stack(labels_list, dim=0)
#         # label_weights: (batch_size, num_classes)
#         label_weights = torch.stack(label_weights_list, dim=0)
#         # Concatenate mask targets: (total_positive, H, W)
#         mask_targets = torch.cat(mask_targets_list, dim=0)
#         # mask_weights: (batch_size, num_classes)
#         mask_weights = torch.stack(mask_weights_list, dim=0)

#         # Flatten classification predictions and targets.
#         cls_scores = cls_scores.flatten(0, 1)  # (batch_size * num_classes, cls_out_channels)
#         labels = labels.flatten(0, 1)          # (batch_size * num_classes,)
#         label_weights = label_weights.flatten(0, 1)

#         class_weight = cls_scores.new_tensor(self.class_weight)

#         loss_cls = self.loss_cls(
#             cls_scores,
#             labels,
#             label_weights,
#             avg_factor=class_weight[labels].sum())

#         # Ensure at least one positive sample.
#         num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
#         num_total_masks = max(num_total_masks, 1)

#         # Extract positive mask predictions.
#         mask_preds = mask_preds[mask_weights > 0]

#         if mask_targets.shape[0] == 0:
#             # No positive matches; return zero losses.
#             loss_dice = mask_preds.sum()
#             loss_mask = mask_preds.sum()
#             return loss_cls, loss_mask, loss_dice

#         # Get uncertain point coordinates for efficient mask loss computation.
#         with torch.no_grad():
#             points_coords = get_uncertain_point_coords_with_randomness(
#                 mask_preds.unsqueeze(1), None, self.num_points,
#                 self.oversample_ratio, self.importance_sample_ratio)
#             # Sample ground truth masks at these points.
#             mask_point_targets = point_sample(mask_targets.unsqueeze(1).float(),
#                                               points_coords).squeeze(1)
#         # Sample mask predictions at these points.
#         mask_point_preds = point_sample(mask_preds.unsqueeze(1), points_coords).squeeze(1)

#         # Compute dice loss.
#         loss_dice = self.loss_dice(
#             mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

#         # Compute mask loss (e.g., binary cross entropy).
#         mask_point_preds = mask_point_preds.reshape(-1, 1)
#         mask_point_targets = mask_point_targets.reshape(-1)
#         loss_mask = self.loss_mask(
#             mask_point_preds,
#             mask_point_targets,
#             avg_factor=num_total_masks * self.num_points)

#         return loss_cls, loss_mask, loss_dice

#     @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
#     def loss(self, all_cls_scores, all_mask_preds, gt_labels_list, gt_masks_list, img_metas):
#         """
#         Compute losses for all decoder layers.

#         Args:
#             all_cls_scores (Tensor): Classification logits for all decoder layers.
#                 Shape: (num_decoder_layers, batch_size, num_queries, cls_out_channels)
#             all_mask_preds (Tensor): Mask predictions for all decoder layers.
#                 Shape: (num_decoder_layers, batch_size, num_queries, H, W)
#             gt_labels_list (list[Tensor]): Ground truth labels for each image.
#             gt_masks_list (list[Tensor]): Ground truth masks for each image.
#             img_metas (list[dict]): Image metadata.

#         Returns:
#             dict: Dictionary containing loss components.
#         """
#         num_dec_layers = len(all_cls_scores)
#         # Replicate ground truth for each decoder layer.
#         all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
#         all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
#         img_metas_list = [img_metas for _ in range(num_dec_layers)]

#         # Compute losses for each decoder layer.
#         losses_cls, losses_mask, losses_dice = multi_apply(
#             self.loss_single, all_cls_scores, all_mask_preds,
#             all_gt_labels_list, all_gt_masks_list, img_metas_list)

#         loss_dict = dict()
#         # Loss from the final decoder layer.
#         loss_dict['loss_cls'] = losses_cls[-1]
#         loss_dict['loss_mask'] = losses_mask[-1]
#         loss_dict['loss_dice'] = losses_dice[-1]
#         # Losses from intermediate decoder layers.
#         for num_dec_layer, (loss_cls_i, loss_mask_i, loss_dice_i) in enumerate(
#                 zip(losses_cls[:-1], losses_mask[:-1], losses_dice[:-1])):
#             loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
#             loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
#             loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
#         return loss_dict

#     def forward_head(self, decoder_out, mask_feature, attn_mask_target_size, get_similarity=False):
#         """
#         Forward pass for the head (called after each transformer decoder layer).

#         Args:
#             decoder_out (Tensor): Output from the transformer decoder.
#                 Shape: (num_queries, batch_size, c)
#             mask_feature (Tensor): Features used for mask prediction.
#                 Shape: (batch_size, c, H, W)
#             attn_mask_target_size (tuple[int, int]): Target size for the attention mask (e.g., (H', W')).
#             get_similarity (bool): If True, also return a similarity map.

#         Returns:
#             tuple: Either (cls_pred, mask_pred, attn_mask) or (cls_pred, sim, attn_mask) if get_similarity is True.
#             - cls_pred: Classification logits. Shape: (batch_size, num_queries, cls_out_channels)
#             - mask_pred: Mask predictions. Shape: (batch_size, num_queries, H, W)
#             - attn_mask: Boolean attention mask.
#                 Shape: (batch_size * num_heads, num_queries, H', W')
#             - sim (optional): Similarity map. Shape: (batch_size, num_queries, H, W)
#         """
#         # Apply post-normalization.
#         decoder_out = self.transformer_decoder.post_norm(decoder_out)
#         # Transpose to (batch_size, num_queries, c)
#         decoder_out = decoder_out.transpose(0, 1)
#         # Classification prediction.
#         cls_pred = self.cls_embed(decoder_out)  # e.g., (B, num_queries, cls_out_channels)
#         # Generate mask embeddings.
#         mask_embed = self.mask_embed(decoder_out)  # (B, num_queries, c)
#         # Compute mask predictions via dot-product with mask features.
#         # Resulting shape: (B, num_queries, H, W)
#         mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
#         # Resize mask_pred to the target attention mask size.
#         attn_mask = F.interpolate(mask_pred, attn_mask_target_size, mode='bilinear', align_corners=False)
#         # Flatten spatial dimensions and repeat for each head.
#         attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat((1, self.num_heads, 1, 1)).flatten(0, 1)
#         # Convert to boolean mask using a sigmoid threshold.
#         attn_mask = (attn_mask.sigmoid() < 0.5).detach()

#         if not get_similarity:
#             return cls_pred, mask_pred, attn_mask
#         else:
#             # Optionally compute a similarity map.
#             mask_feature = F.normalize(mask_feature, dim=1, p=2)
#             mask_embed = F.normalize(mask_embed, dim=-1, p=2)
#             sim = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
#             sim = (sim + 1.) / 2.
#             return cls_pred, sim, attn_mask

#     def forward(self, feats, texts, img_metas, return_mask_features=False, get_similarity=False, seg_mask=None):
#         """
#         Forward function for both training and testing.

#         Args:
#             feats (list[Tensor]): Multi-scale features from the backbone.
#                 Each tensor shape: (B, C, H, W)
#             texts (Tensor): Text embeddings.
#             img_metas (list[dict]): List of image metadata.
#             return_mask_features (bool): If True, also return mask features.
#             get_similarity (bool): If True, return a similarity map instead of mask predictions.
#             seg_mask (Tensor, optional): Semantic segmentation mask used for certain merging strategies.
#                 (e.g., for the 'matching' method)

#         Returns:
#             tuple: (cls_pred_list, mask_pred_list) or (cls_pred_list, mask_pred_list, mask_features)
#             - cls_pred_list: List of classification logits from each decoder layer.
#                 Each tensor shape: (B, num_queries, cls_out_channels)
#             - mask_pred_list: List of mask predictions from each decoder layer.
#                 Each tensor shape: (B, num_queries, H, W)
#         """
#         batch_size = len(img_metas)
#         # Obtain mask features and multi-scale memory from the pixel decoder.
#         # mask_features: (B, C, H, W)
#         # multi_scale_memorys: list of feature maps (one per transformer level)
#         mask_features, multi_scale_memorys = self.pixel_decoder(feats, texts)

#         # Prepare decoder inputs and positional encodings for each transformer feature level.
#         decoder_inputs = []
#         decoder_positional_encodings = []
#         for i in range(self.num_transformer_feat_level):
#             decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
#             # Reshape: (B, C, H, W) -> (H*W, B, C)
#             decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
#             # Add level-specific embedding.
#             level_embed = self.level_embed.weight[i].view(1, 1, -1)
#             decoder_input = decoder_input + level_embed
#             # Generate positional encoding.
#             mask = decoder_input.new_zeros((batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool)
#             decoder_positional_encoding = self.decoder_positional_encoding(mask)
#             # Reshape: (B, C, H, W) -> (H*W, B, C)
#             decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(2, 0, 1)
#             decoder_inputs.append(decoder_input)
#             decoder_positional_encodings.append(decoder_positional_encoding)

#         # Generate query features from text.
#         # After text projection and permutation: (num_queries, B, feat_channels)
#         query_feat = self.text_proj(texts).permute(1, 0, 2)
#         # Obtain fixed query embeddings and repeat for each batch.
#         # Shape: (num_queries, B, feat_channels)
#         query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))

#         cls_pred_list = []
#         mask_pred_list = []

#         # Initial head prediction from the query features.
#         orig_cls_pred, orig_mask_pred, attn_mask = self.forward_head(
#             query_feat, mask_features, multi_scale_memorys[0].shape[-2:], get_similarity=get_similarity)

#         # M: number of learnable queries = num_queries - num_classes
#         M = self.M

#         # Merge fixed queries (first num_classes) and learnable queries (remaining M+1)
#         if seg_mask is None:
#             # In inference mode, select merging strategy based on mask_pred_method.
#             if self.mask_pred_method == 'matching':
#                 fixed_cls = orig_cls_pred[:, :-(M+1), :]
#                 fixed_mask = orig_mask_pred[:, :-(M+1), :, :]
#                 learnable_cls = orig_cls_pred[:, -(M+1):, :]    # e.g., shape (B, M+1, num_cls)
#                 learnable_mask = orig_mask_pred[:, -(M+1):, :, :]  # e.g., shape (B, M+1, H, W)
#                 # Compute confidence scores.
#                 learnable_cls_conf = F.softmax(learnable_cls, dim=-1)
#                 scores, labels = learnable_cls_conf.max(dim=-1)
#                 # Apply a threshold to select valid queries.
#                 keep = (labels.ne(self.num_classes)) & (scores > 0.95) & (labels < 11) & (labels > 1)
#                 weighted_learnable_mask = learnable_mask.sigmoid() * scores.unsqueeze(-1).unsqueeze(-1)
#                 weighted_learnable_mask = weighted_learnable_mask * keep.unsqueeze(-1).unsqueeze(-1).float()
#                 final_learnable_mask = weighted_learnable_mask.mean(dim=1, keepdim=True)
#                 final_learnable_cls  = learnable_cls.mean(dim=1, keepdim=True)

#                 cls_pred = torch.cat([fixed_cls, final_learnable_cls], dim=1)
#                 mask_pred = torch.cat([fixed_mask, final_learnable_mask], dim=1)
#             elif self.mask_pred_method == 'mean':
#                 mask_pred = torch.cat([
#                     orig_mask_pred[:, :-(M+1), :, :],
#                     torch.mean(orig_mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)
#                 ], dim=1)
#                 cls_pred = torch.cat([
#                     orig_cls_pred[:, :-(M+1), :],
#                     torch.mean(orig_cls_pred[:, -(M+1):, :], dim=1, keepdim=True)
#                 ], dim=1)
#             elif self.mask_pred_method == 'max':
#                 # For 'max' method, use the original predictions without merging.
#                 mask_pred = orig_mask_pred
#                 cls_pred = orig_cls_pred
#             elif self.mask_pred_method == 'ood_scoring':
#                 fixed_cls = orig_cls_pred[:, :-(M+1), :]
#                 fixed_mask = orig_mask_pred[:, :-(M+1), :, :]
#                 learnable_cls = orig_cls_pred[:, -(M+1):, :]
#                 learnable_mask = orig_mask_pred[:, -(M+1):, :, :]

#                 learnable_cls_prob = F.softmax(learnable_cls, dim=-1)
#                 cls19_prob = learnable_cls_prob[:, :, 19]
#                 cls19_prob = F.softmax(cls19_prob, dim=1)

#                 final_learnable_mask = (learnable_mask * cls19_prob.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
#                 final_learnable_cls = (learnable_cls * cls19_prob.unsqueeze(-1)).sum(dim=1, keepdim=True)

#                 cls_pred = torch.cat([fixed_cls, final_learnable_cls], dim=1)
#                 mask_pred = torch.cat([fixed_mask, final_learnable_mask], dim=1)
#             elif self.mask_pred_method == 'ood_scoring_contrastive':
#                 mask_pred = orig_mask_pred
#                 cls_pred = orig_cls_pred
#             else:
#                 mask_pred = orig_mask_pred
#                 cls_pred = orig_cls_pred
#         else:
#             # When seg_mask is provided (e.g., during training), use the corresponding merge strategy.
#             if self.mask_pred_method == 'mean':
#                 mask_pred = torch.cat([
#                     orig_mask_pred[:, :-(M+1), :, :],
#                     torch.mean(orig_mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)
#                 ], dim=1)
#                 cls_pred = torch.cat([
#                     orig_cls_pred[:, :-(M+1), :],
#                     torch.mean(orig_cls_pred[:, -(M+1):, :], dim=1, keepdim=True)
#                 ], dim=1)
#             elif self.mask_pred_method == 'max':
#                 mask_pred = torch.cat([
#                     orig_mask_pred[:, :-(M+1), :, :],
#                     torch.max(orig_mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)[0]
#                 ], dim=1)
#                 cls_pred = torch.cat([
#                     orig_cls_pred[:, :-(M+1), :],
#                     torch.mean(orig_cls_pred[:, -(M+1):, :], dim=1, keepdim=True)
#                 ], dim=1)
#             elif self.mask_pred_method == 'matching':
#                 target_size = orig_mask_pred.shape[-2:]
#                 gt_mask = (seg_mask == 19).float()
#                 learnable_channels = orig_mask_pred[:, -(M+1):, :, :]
#                 gt_mask = F.interpolate(gt_mask, size=learnable_channels.shape[-2:], mode='nearest')
#                 matching_scores = torch.sum(learnable_channels * gt_mask, dim=(2, 3)) / (torch.sum(gt_mask, dim=(2, 3)) + 1e-6)
#                 best_channel_idx = matching_scores.argmax(dim=1)
#                 best_channel = []
#                 for i in range(orig_mask_pred.shape[0]):
#                     best_channel.append(learnable_channels[i, best_channel_idx[i], :, :].unsqueeze(0))
#                 best_channel = torch.cat(best_channel, dim=0).unsqueeze(1)
#                 mask_pred = torch.cat([orig_mask_pred[:, :-(M+1), :, :], best_channel], dim=1)
#                 cls_pred = torch.cat([orig_cls_pred[:, :-(M+1), :],
#                                       torch.mean(orig_cls_pred[:, -(M+1):, :], dim=1, keepdim=True)], dim=1)
#             elif self.mask_pred_method == 'ood_scoring':
#                 fixed_cls = orig_cls_pred[:, :-(M+1), :]
#                 fixed_mask = orig_mask_pred[:, :-(M+1), :, :]
#                 learnable_cls = orig_cls_pred[:, -(M+1):, :]
#                 learnable_mask = orig_mask_pred[:, -(M+1):, :, :]

#                 learnable_cls_prob = F.softmax(learnable_cls, dim=-1)
#                 cls19_prob = learnable_cls_prob[:, :, 19]
#                 cls19_prob = F.softmax(cls19_prob, dim=1)

#                 final_learnable_mask = (learnable_mask * cls19_prob.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
#                 final_learnable_cls = (learnable_cls * cls19_prob.unsqueeze(-1)).sum(dim=1, keepdim=True)

#                 cls_pred = torch.cat([fixed_cls, final_learnable_cls], dim=1)
#                 mask_pred = torch.cat([fixed_mask, final_learnable_mask], dim=1)
#             elif self.mask_pred_method == 'ood_scoring_contrastive':
#                 fixed_cls = orig_cls_pred[:, :-(M+1), :]
#                 fixed_mask = orig_mask_pred[:, :-(M+1), :, :]
#                 learnable_cls = orig_cls_pred[:, -(M+1):, :]
#                 learnable_mask = orig_mask_pred[:, -(M+1):, :, :]

#                 learnable_cls_prob = F.softmax(learnable_cls, dim=-1)
#                 cls19_prob = learnable_cls_prob[:, :, 19]
#                 ood_contrastive_weights = F.softmax(cls19_prob, dim=1)
#                 max_cls19_idx = torch.argmax(cls19_prob, dim=1, keepdim=True)
#                 final_learnable_mask = torch.gather(learnable_mask, 1, max_cls19_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, learnable_mask.shape[2], learnable_mask.shape[3]))
#                 final_learnable_cls = torch.gather(learnable_cls, 1, max_cls19_idx.unsqueeze(-1).repeat(1, 1, learnable_cls.shape[2]))
#                 final_learnable_cls = final_learnable_cls * torch.gather(ood_contrastive_weights, 1, max_cls19_idx).unsqueeze(-1)
#                 cls_contrastive_weights = F.softmax(orig_cls_pred, dim=-1)
#                 fixed_cls_weights = cls_contrastive_weights[:, :-(M+1), :]
#                 fixed_cls = fixed_cls * (1 - fixed_cls_weights[:, :, 19].unsqueeze(-1))
#                 cls_pred = torch.cat([fixed_cls, final_learnable_cls], dim=1)
#                 mask_pred = torch.cat([fixed_mask, final_learnable_mask], dim=1)
#             else:
#                 mask_pred = orig_mask_pred
#                 cls_pred = orig_cls_pred

#         cls_pred_list.append(cls_pred)
#         mask_pred_list.append(mask_pred)

#         # Process additional transformer decoder layers.
#         for i in range(self.num_transformer_decoder_layers):
#             level_idx = i % self.num_transformer_feat_level
#             # If the attention mask is entirely True, reset it to False.
#             attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

#             layer = self.transformer_decoder.layers[i]
#             attn_masks = [attn_mask, None]
#             query_feat = layer(
#                 query=query_feat,
#                 key=decoder_inputs[level_idx],
#                 value=decoder_inputs[level_idx],
#                 query_pos=query_embed,
#                 key_pos=decoder_positional_encodings[level_idx],
#                 attn_masks=attn_masks,
#                 query_key_padding_mask=None,
#                 key_padding_mask=None
#             )
#             orig_cls_pred, orig_mask_pred, attn_mask = self.forward_head(
#                 query_feat, mask_features,
#                 multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:], get_similarity=get_similarity
#             )

#             if seg_mask is None:
#                 if self.mask_pred_method == 'matching':
#                     fixed_cls = orig_cls_pred[:, :-(M+1), :]
#                     fixed_mask = orig_mask_pred[:, :-(M+1), :, :]
#                     learnable_cls = orig_cls_pred[:, -(M+1):, :]
#                     learnable_mask = orig_mask_pred[:, -(M+1):, :, :]

#                     learnable_cls_conf = F.softmax(learnable_cls, dim=-1)
#                     scores, labels = learnable_cls_conf.max(dim=-1)
#                     keep = (labels.ne(self.num_classes)) & (scores > 0.95) & (labels < 11) & (labels > 1)
#                     weighted_learnable_mask = learnable_mask.sigmoid() * scores.unsqueeze(-1).unsqueeze(-1)
#                     weighted_learnable_mask = weighted_learnable_mask * keep.unsqueeze(-1).unsqueeze(-1).float()
#                     final_learnable_mask = weighted_learnable_mask.mean(dim=1, keepdim=True)
#                     final_learnable_cls = learnable_cls.mean(dim=1, keepdim=True)

#                     cls_pred = torch.cat([fixed_cls, final_learnable_cls], dim=1)
#                     mask_pred = torch.cat([fixed_mask, final_learnable_mask], dim=1)
#                 else:
#                     if self.mask_pred_method == 'mean':
#                         mask_pred = torch.cat([
#                             orig_mask_pred[:, :-(M+1), :, :],
#                             torch.mean(orig_mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)
#                         ], dim=1)
#                         cls_pred = torch.cat([
#                             orig_cls_pred[:, :-(M+1), :],
#                             torch.mean(orig_cls_pred[:, -(M+1):, :], dim=1, keepdim=True)
#                         ], dim=1)
#                     elif self.mask_pred_method == 'max':
#                         mask_pred = orig_mask_pred
#                         cls_pred = orig_cls_pred
#                     elif self.mask_pred_method == 'ood_scoring':
#                         fixed_cls = orig_cls_pred[:, :-(M+1), :]
#                         fixed_mask = orig_mask_pred[:, :-(M+1), :, :]
#                         learnable_cls = orig_cls_pred[:, -(M+1):, :]
#                         learnable_mask = orig_mask_pred[:, -(M+1):, :, :]

#                         learnable_cls_prob = F.softmax(learnable_cls, dim=-1)
#                         cls19_prob = learnable_cls_prob[:, :, 19]
#                         cls19_prob = F.softmax(cls19_prob, dim=1)

#                         final_learnable_mask = (learnable_mask * cls19_prob.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
#                         final_learnable_cls = (learnable_cls * cls19_prob.unsqueeze(-1)).sum(dim=1, keepdim=True)

#                         cls_pred = torch.cat([fixed_cls, final_learnable_cls], dim=1)
#                         mask_pred = torch.cat([fixed_mask, final_learnable_mask], dim=1)
#                     elif self.mask_pred_method == 'ood_scoring_contrastive':
#                         mask_pred = orig_mask_pred
#                         cls_pred = orig_cls_pred
#                     else:
#                         mask_pred = orig_mask_pred
#                         cls_pred = orig_cls_pred
#             else:
#                 if self.mask_pred_method == 'mean':
#                     mask_pred = torch.cat([
#                         orig_mask_pred[:, :-(M+1), :, :],
#                         torch.mean(orig_mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)
#                     ], dim=1)
#                     cls_pred = torch.cat([
#                         orig_cls_pred[:, :-(M+1), :],
#                         torch.mean(orig_cls_pred[:, -(M+1):, :], dim=1, keepdim=True)
#                     ], dim=1)
#                 elif self.mask_pred_method == 'max':
#                     mask_pred = torch.cat([
#                         orig_mask_pred[:, :-(M+1), :, :],
#                         torch.max(orig_mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)[0]
#                     ], dim=1)
#                     cls_pred = torch.cat([
#                         orig_cls_pred[:, :-(M+1), :],
#                         torch.mean(orig_cls_pred[:, -(M+1):, :], dim=1, keepdim=True)
#                     ], dim=1)
#                 elif self.mask_pred_method == 'matching':
#                     target_size = orig_mask_pred.shape[-2:]
#                     gt_mask = (seg_mask == 19).float()
#                     learnable_channels = orig_mask_pred[:, -(M+1):, :, :]
#                     gt_mask = F.interpolate(gt_mask, size=learnable_channels.shape[-2:], mode='nearest')
#                     matching_scores = torch.sum(learnable_channels * gt_mask, dim=(2, 3)) / (torch.sum(gt_mask, dim=(2, 3)) + 1e-6)
#                     best_channel_idx = matching_scores.argmax(dim=1)
#                     best_channel = []
#                     for j in range(orig_mask_pred.shape[0]):
#                         best_channel.append(learnable_channels[j, best_channel_idx[j], :, :].unsqueeze(0))
#                     best_channel = torch.cat(best_channel, dim=0).unsqueeze(1)
#                     mask_pred = torch.cat([orig_mask_pred[:, :-(M+1), :, :], best_channel], dim=1)
#                     cls_pred = torch.cat([orig_cls_pred[:, :-(M+1), :],
#                                           torch.mean(orig_cls_pred[:, -(M+1):, :], dim=1, keepdim=True)], dim=1)
#                 elif self.mask_pred_method == 'ood_scoring':
#                     fixed_cls = orig_cls_pred[:, :-(M+1), :]
#                     fixed_mask = orig_mask_pred[:, :-(M+1), :, :]
#                     learnable_cls = orig_cls_pred[:, -(M+1):, :]
#                     learnable_mask = orig_mask_pred[:, -(M+1):, :, :]

#                     learnable_cls_prob = F.softmax(learnable_cls, dim=-1)
#                     cls19_prob = learnable_cls_prob[:, :, 19]
#                     cls19_prob = F.softmax(cls19_prob, dim=1)

#                     final_learnable_mask = (learnable_mask * cls19_prob.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
#                     final_learnable_cls = (learnable_cls * cls19_prob.unsqueeze(-1)).sum(dim=1, keepdim=True)

#                     cls_pred = torch.cat([fixed_cls, final_learnable_cls], dim=1)
#                     mask_pred = torch.cat([fixed_mask, final_learnable_mask], dim=1)
#                 elif self.mask_pred_method == 'ood_scoring_contrastive':
#                     fixed_cls = orig_cls_pred[:, :-(M+1), :]
#                     fixed_mask = orig_mask_pred[:, :-(M+1), :, :]
#                     learnable_cls = orig_cls_pred[:, -(M+1):, :]
#                     learnable_mask = orig_mask_pred[:, -(M+1):, :, :]

#                     learnable_cls_prob = F.softmax(learnable_cls, dim=-1)
#                     cls19_prob = learnable_cls_prob[:, :, 19]
#                     ood_contrastive_weights = F.softmax(cls19_prob, dim=1)
#                     max_cls19_idx = torch.argmax(cls19_prob, dim=1, keepdim=True)
#                     final_learnable_mask = torch.gather(learnable_mask, 1, max_cls19_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, learnable_mask.shape[2], learnable_mask.shape[3]))
#                     final_learnable_cls = torch.gather(learnable_cls, 1, max_cls19_idx.unsqueeze(-1).repeat(1, 1, learnable_cls.shape[2]))
#                     final_learnable_cls = final_learnable_cls * torch.gather(ood_contrastive_weights, 1, max_cls19_idx).unsqueeze(-1)
#                     cls_contrastive_weights = F.softmax(orig_cls_pred, dim=-1)
#                     fixed_cls_weights = cls_contrastive_weights[:, :-(M+1), :]
#                     remaining_fixed_cls = fixed_cls * (1 - fixed_cls_weights[:, :, 19].unsqueeze(-1))
#                     background_cls = remaining_fixed_cls.mean(dim=1, keepdim=True)
#                     fixed_cls = fixed_cls * (1 - fixed_cls_weights[:, :, 19].unsqueeze(-1))
#                     cls_pred = torch.cat([fixed_cls, final_learnable_cls], dim=1)
#                     mask_pred = torch.cat([fixed_mask, final_learnable_mask], dim=1)
#                 else:
#                     mask_pred = orig_mask_pred
#                     cls_pred = orig_cls_pred

#             cls_pred_list.append(cls_pred)
#             mask_pred_list.append(mask_pred)

#         if return_mask_features:
#             return cls_pred_list, mask_pred_list, mask_features
#         else:
#             return cls_pred_list, mask_pred_list

#     def forward_train(self, x, texts, img_metas, gt_semantic_seg, train_cfg,
#                       gt_labels, gt_masks):
#         """
#         Forward function for training.

#         Args:
#             x (list[Tensor]): Multi-level features from the backbone.
#                 Each tensor shape: (B, C, H, W)
#             texts (Tensor): Text embeddings.
#             img_metas (list[dict]): Image metadata.
#             gt_semantic_seg (Tensor): Ground truth semantic segmentation.
#                 Shape: (B, H, W)
#             train_cfg (dict): Training configuration.
#             gt_labels (list[Tensor]): Ground truth instance labels.
#                 Each tensor shape: (num_gts,)
#             gt_masks (list[Tensor]): Ground truth instance masks.
#                 Each tensor shape: (num_gts, H, W)

#         Returns:
#             dict: A dictionary of loss components.
#         """
#         # Forward pass with segmentation mask (seg_mask) used for merging queries.
#         all_cls_scores, all_mask_preds = self(x, texts, img_metas, seg_mask=gt_semantic_seg)
#         losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks, img_metas)
#         return losses

#     def forward_test(self, inputs, texts, img_metas, test_cfg):
#         """
#         Test segmentation without test-time augmentation.

#         Only the outputs of the last decoder layer are used.

#         Args:
#             inputs (list[Tensor]): Multi-level features from the backbone.
#             texts (Tensor): Text embeddings.
#             img_metas (list[dict]): Image metadata.
#             test_cfg (dict): Testing configuration.

#         Returns:
#             Tensor: Predicted segmentation mask.
#                 Shape: (B, num_classes, H, W)
#         """
#         all_cls_scores, all_mask_preds = self(inputs, texts, img_metas)
#         cls_score, mask_pred = all_cls_scores[-1], all_mask_preds[-1]
#         ori_h, ori_w, _ = img_metas[0]['ori_shape']

#         if self.mask_pred_method == 'max':
#             M = self.M
#             # For the 'max' method, select fixed query predictions (first num_classes)
#             # and choose one learnable query (e.g., index 21) to merge.
#             mask_pred = torch.cat([
#                 mask_pred[:, :-(M+1), :, :],
#                 mask_pred[:, 21:22, :, :]  # Example: selecting the 22nd query.
#             ], dim=1)
#             cls_score = torch.cat([
#                 cls_score[:, :-(M+1), :],
#                 cls_score[:, 21:22, :]  # Corresponding classification logits.
#             ], dim=1)

#             # Semantic inference: apply softmax for classification and sigmoid for mask.
#             cls_score = F.softmax(cls_score, dim=-1)[..., :-1]
#             mask_pred = mask_pred.sigmoid()
#             seg_mask = torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred)
#         elif self.mask_pred_method == 'ood_scoring_contrastive':
#             M = self.M
#             cls_score = F.softmax(cls_score, dim=-1)[..., :-1]
#             mask_pred = mask_pred.sigmoid()
#             id_cls_score = cls_score[:, :-(M+1), :]
#             id_mask_pred = mask_pred[:, :-(M+1), :, :]
#             ood_cls_score = cls_score[:, -(M+1):, :]
#             ood_mask_pred = mask_pred[:, -(M+1):, :, :]
#             seg_mask_id = torch.einsum('bqc,bqhw->bchw', id_cls_score, id_mask_pred)
#             seg_mask_ood = torch.einsum('bqc,bqhw->bchw', ood_cls_score, ood_mask_pred)
#             seg_mask = torch.cat([seg_mask_id[:, :19, :, :], seg_mask_ood[:, 19:, :, :]], dim=1)
#         else:
#             # Default inference.
#             cls_score = F.softmax(cls_score, dim=-1)[..., :-1]
#             mask_pred = mask_pred.sigmoid()
#             seg_mask = torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred)
#         return seg_mask

#     def forward_inference(self, inputs, texts, img_metas, test_cfg):
#         """
#         Inference function.

#         Args:
#             inputs (list[Tensor]): Multi-level features from the backbone.
#             texts (Tensor): Text embeddings.
#             img_metas (list[dict]): Image metadata.
#             test_cfg (dict): Testing configuration.

#         Returns:
#             tuple: (all_mask_preds, mask_features)
#         """
#         all_cls_scores, all_mask_preds, mask_features = \
#             self(inputs, texts, img_metas, return_mask_features=True)
#         return all_mask_preds, mask_features

        
############################################################################################################

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, build_plugin_layer, caffe2_xavier_init, ConvModule
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.ops import point_sample
from mmcv.runner import ModuleList, force_fp32
from mmseg.models.builder import HEADS, build_loss
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from ...core import build_sampler, multi_apply, reduce_mean
from ..builder import build_assigner
from ..utils import get_uncertain_point_coords_with_randomness

@HEADS.register_module()
class tqdmHead(BaseDecodeHead):

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 num_queries=100,
                 num_transformer_feat_level=3,
                 pixel_decoder=None,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 loss_cls=None,
                 loss_mask=None,
                 loss_dice=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 text_proj=None,
                 **kwargs):
        super(tqdmHead, self).__init__(
            in_channels=in_channels,
            channels=feat_channels,
            num_classes=(num_things_classes + num_stuff_classes),
            init_cfg=init_cfg,
            input_transform='multiple_select',
            **kwargs)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers. \
            attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.transformerlayers. \
                   attn_cfgs[0].num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)

        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels,
            text_proj=text_proj) #
        self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]

        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = build_positional_encoding(
            positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.text_proj = nn.Sequential(
            nn.Linear(text_proj.text_in_dim, text_proj.text_out_dim), nn.ReLU(inplace=True),
            nn.Linear(text_proj.text_out_dim, text_proj.text_out_dim))
        # self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))
        self.conv_seg = None # fix a bug here (conv_seg is not used)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.class_weight = loss_cls.class_weight
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

        ##### #$#$#$#$#$ method 4

        # self.loss_contrastive = build_loss({'type': 'MaskContrastiveLoss'})
        # self.ood_score_conv = nn.Conv2d(in_channels=20, out_channels=1, kernel_size=1)
        
        ##### #$#$#$#$#$#

        ##### #$#$#$#$#$#
        # self.mask_pred_method = 'ood_scoring_contrastive'# 'matching'
        self.mask_pred_method = 'max'
        self.M = self.num_queries - self.num_classes
        ##### #$#$#$#$#$#

    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                    gt_masks_list, img_metas):
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape [num_queries,
                cls_out_channels].
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape [num_queries, h, w].
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels of all images.
                    Each with shape [num_queries, ].
                - label_weights_list (list[Tensor]): Label weights of all
                    images.Each with shape [num_queries, ].
                - mask_targets_list (list[Tensor]): Mask targets of all images.
                    Each with shape [num_queries, h, w].
                - mask_weights_list (list[Tensor]): Mask weights of all images.
                    Each with shape [num_queries, ].
                - num_total_pos (int): Number of positive samples in all
                    images.
                - num_total_neg (int): Number of negative samples in all
                    images.
        """
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list,
                                      mask_preds_list, gt_labels_list,
                                      gt_masks_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks,
                           img_metas):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (num_gts, ).
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (num_gts, h, w).
            img_metas (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
        """
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                               1)).squeeze(1)

        # assign and sample
        assign_result = self.assigner.assign(cls_score, mask_points_pred,
                                             gt_labels, gt_points_masks,
                                             img_metas)
        sampling_result = self.sampler.sample(assign_result, mask_pred,
                                              gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target

        # original
        # labels = gt_labels.new_full((self.num_queries, ),
        #                             self.num_classes,
        #                             dtype=torch.long)
        #$#$#$#$#$#$#$ start
        labels = gt_labels.new_full((self.num_classes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        #$#$#$#$#$#$#$ end
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]

        # original
        # label_weights = gt_labels.new_ones((self.num_queries, ))
        #$#$#$#$#$#$#$ start
        label_weights = gt_labels.new_ones((self.num_classes, ))
        #$#$#$#$#$#$#$ end

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        # original
        # mask_weights = mask_pred.new_zeros((self.num_queries, ))
        #$#$#$#$#$#$#$ start
        mask_weights = mask_pred.new_zeros((self.num_classes, ))
        #$#$#$#$#$#$#$ end
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds)

    def loss_single(self, cls_scores, mask_preds, gt_labels_list,
                    gt_masks_list, img_metas):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (num_gts, ).
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (num_gts, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

        # print("len(gt_labels_list)",len(gt_labels_list)) #$#$#$#$#$
        # print("gt_labels_list[0].shape",gt_labels_list[0].shape) #$#$#$#$#$
        # print("gt_labels_list[1].shape",gt_labels_list[1].shape) #$#$#$#$#$
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,
         num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list,
                                           gt_labels_list, gt_masks_list,
                                           img_metas)
        # print("len(labels_list)",len(labels_list)) #$#$#$#$#$
        # print("labels_list[0].shape",labels_list[0].shape) #$#$#$#$#$
        # print("labels_list[1].shape",labels_list[1].shape) #$#$#$#$#$
        
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # print("cls_scores.shape",cls_scores.shape) #$#$#$#$#$
        # print("labels.shape",labels.shape) #$#$#$#$#$
        
        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)
        
        class_weight = cls_scores.new_tensor(self.class_weight)

        # print("cls_scores.shape",cls_scores.shape) #$#$#$#$#$
        # print("labels.shape",labels.shape) #$#$#$#$#$
        # assert False
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        # print("mask_preds.shape", mask_preds.shape) #$#$#$#$#$
        # print("mask_weights.shape", mask_weights.shape) #$#$#$#$#$
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)


        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # print("mask_preds_list[0].shape",mask_preds_list[0].shape)
        # print("mask_targets.shape",mask_targets.shape)
        # print("mask_point_preds.shape",mask_point_preds.shape)
        # print("mask_point_targets.shape",mask_point_targets.shape)
        # assert False
        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1,1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)


        # loss_contrastive = self.loss_contrastive(
        #     mask_point_preds,  # (N, P)
        #     mask_point_targets,
        #     avg_factor=num_total_masks  # or num_total_masks * num_points, etc.
        # )
        # assert False
        return loss_cls, loss_mask, loss_dice

    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss(self, all_cls_scores, all_mask_preds, gt_labels_list,
             gt_masks_list, img_metas):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape [num_decoder, batch_size, num_queries,
                cls_out_channels].
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape [num_decoder, batch_size, num_queries, h, w].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (n, ). n is the sum of number of stuff type
                and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image with
                shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        # print("gt_labels_list[0].shape",gt_labels_list[0].shape) #$#$#$#$#$#$
        # print("gt_labels_list[1].shape",gt_labels_list[1].shape) #$#$#$#$#$#$
        losses_cls, losses_mask, losses_dice = multi_apply(
            self.loss_single, all_cls_scores, all_mask_preds,
            all_gt_labels_list, all_gt_masks_list, img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1
        return loss_dict

    def forward_head(self, decoder_out, mask_feature, attn_mask_target_size, get_similarity=False):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

            - cls_pred (Tensor): Classification scores in shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred (Tensor): Mask scores in shape \
                (batch_size, num_queries,h, w).
            - attn_mask (Tensor): Attention mask in shape \
                (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        # shape (num_queries, batch_size, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        if get_similarity is False:
            return cls_pred, mask_pred, attn_mask
        else:
            mask_feature = F.normalize(mask_feature, dim=1, p=2)
            mask_embed = F.normalize(mask_embed, dim=-1, p=2)
            sim = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
            sim = (sim + 1.) / 2.
            return cls_pred, sim, attn_mask
    
    # def forward(self, feats, texts, img_metas, return_mask_features=False, get_similarity=False, seg_mask=None):
    #     """Forward function.
    
    #     Args:
    #         feats (list[Tensor]): Multi scale Features from the upstream network, each is a 4D-tensor.
    #         img_metas (list[dict]): List of image information.
    #         seg_mask (Tensor, optional): Semantic segmentation mask. Used for matching method.
        
    #     Returns:
    #         tuple: A tuple contains two elements.
    
    #             - cls_pred_list (list[Tensor]): Classification logits for each decoder layer.
    #               Each is a 3D-tensor with shape (batch_size, num_queries, cls_out_channels).
    #               Note `cls_out_channels` should include background.
    #             - mask_pred_list (list[Tensor]): Mask logits for each decoder layer.
    #               Each with shape (batch_size, num_queries, h, w).
    #     """
    #     batch_size = len(img_metas)
    #     mask_features, multi_scale_memorys = self.pixel_decoder(feats, texts)  # pixel_decoder needs texts!
    #     # multi_scale_memorys (from low resolution to high resolution)
    #     decoder_inputs = []
    #     decoder_positional_encodings = []
    #     for i in range(self.num_transformer_feat_level):
    #         decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
    #         # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
    #         decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
    #         level_embed = self.level_embed.weight[i].view(1, 1, -1)
    #         decoder_input = decoder_input + level_embed
    #         # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
    #         mask = decoder_input.new_zeros((batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool)
    #         decoder_positional_encoding = self.decoder_positional_encoding(mask)
    #         decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(2, 0, 1)
    #         decoder_inputs.append(decoder_input)
    #         decoder_positional_encodings.append(decoder_positional_encoding)
    #     # shape (num_queries, c) -> (num_queries, batch_size, c)
    #     query_feat = self.text_proj(texts).permute(1, 0, 2)
    #     query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))
    
    #     cls_pred_list = []
    #     mask_pred_list = []
    #     cls_pred, mask_pred, attn_mask = self.forward_head(query_feat, mask_features, multi_scale_memorys[0].shape[-2:], get_similarity=get_similarity)
    
    #     #--------------------------------------------
    #     # learnable query로 늘어난 부분을 줄이는 과정
    #     M = self.num_queries - self.num_classes  # M+1: learnable query 개수
    
    #     # cls_pred는 기존대로 평균 방식으로 합침
    #     cls_pred = torch.cat([
    #         cls_pred[:, :-(M+1), :],  # 첫 번째 부분
    #         torch.mean(cls_pred[:, -(M+1):, :], dim=1, keepdim=True)  # 두 번째 부분: M+1개의 채널을 평균
    #     ], dim=1)
    
    #     # $#$#$#$#$#$ 수정 시작: mask_pred 합치는 방식 선택
    #     if seg_mask == None:  # inference 시
    #         mask_pred = torch.cat([
    #             mask_pred[:, :-(M+1), :, :],
    #             torch.max(mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)[0]
    #         ], dim=1)
    #     elif self.mask_pred_method == 'mean':
    #         mask_pred = torch.cat([
    #             mask_pred[:, :-(M+1), :, :],
    #             torch.mean(mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)
    #         ], dim=1)
    #     elif self.mask_pred_method == 'max':
    #         mask_pred = torch.cat([
    #             mask_pred[:, :-(M+1), :, :],
    #             torch.max(mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)[0]
    #         ], dim=1)

    #     elif self.mask_pred_method == 'matching':
    #         # mask_pred의 해상도에 맞게 seg_mask를 리사이즈합니다.
    #         target_size = mask_pred.shape[-2:]  # 예: (128, 128)
    #         gt_mask = (seg_mask == 19).float()    # 원래 shape: [B, 1, H, W] (예: 512×512)
            
    #         # learnable 채널들 (마지막 (M+1) 채널)
    #         learnable_channels = mask_pred[:, -(M+1):, :, :]  # shape: [B, M+1, target_H, target_W]
    #         # 조건 없이 gt_mask의 해상도를 예측 결과와 동일하게 맞춥니다.
    #         gt_mask = F.interpolate(gt_mask, size=learnable_channels.shape[-2:], mode='nearest')

    #         # 각 채널에 대해 gt_mask 영역의 평균 값을 매칭 스코어로 계산
    #         matching_scores = torch.sum(learnable_channels * gt_mask, dim=(2, 3)) / (torch.sum(gt_mask, dim=(2, 3)) + 1e-6)  # shape: [B, M+1]            
    #         best_channel_idx = matching_scores.argmax(dim=1)  # shape: [B]
            
    #         # 각 배치에서 가장 높은 매칭 스코어를 가진 채널 선택
    #         best_channel = []
    #         for i in range(mask_pred.shape[0]):
    #             best_channel.append(learnable_channels[i, best_channel_idx[i], :, :].unsqueeze(0))
    #         best_channel = torch.cat(best_channel, dim=0)  # shape: [B, target_H, target_W]
    #         best_channel = best_channel.unsqueeze(1)        # shape: [B, 1, target_H, target_W]
            
    #         # 기존 채널과 선택된 채널을 합침
    #         mask_pred = torch.cat([
    #             mask_pred[:, :-(M+1), :, :],
    #             best_channel
    #         ], dim=1)

    #     else:
    #         # fallback: 기본 평균 방식
    #         mask_pred = torch.cat([
    #             mask_pred[:, :-(M+1), :, :],
    #             torch.max(mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)[0]
    #         ], dim=1)
    #     # $#$#$#$#$#$ 수정 끝
    #     #--------------------------------------------
    
    #     cls_pred_list.append(cls_pred)
    #     mask_pred_list.append(mask_pred)
    
    #     for i in range(self.num_transformer_decoder_layers):
    #         level_idx = i % self.num_transformer_feat_level
    #         # 만약 attn_mask가 모두 True (즉, 모두 background)인 경우 모두 False로 변경
    #         attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
    
    #         # cross_attn + self_attn
    #         layer = self.transformer_decoder.layers[i]
    #         attn_masks = [attn_mask, None]
    #         query_feat = layer(
    #             query=query_feat,
    #             key=decoder_inputs[level_idx],
    #             value=decoder_inputs[level_idx],
    #             query_pos=query_embed,
    #             key_pos=decoder_positional_encodings[level_idx],
    #             attn_masks=attn_masks,
    #             query_key_padding_mask=None,
    #             key_padding_mask=None
    #         )
    #         cls_pred, mask_pred, attn_mask = self.forward_head(
    #             query_feat, mask_features,
    #             multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:],
    #             get_similarity=get_similarity
    #         )
            
    #         # cls_pred는 동일하게 평균 방식으로 합침
    #         cls_pred = torch.cat([
    #             cls_pred[:, :-(M+1), :],
    #             torch.mean(cls_pred[:, -(M+1):, :], dim=1, keepdim=True)
    #         ], dim=1)
    
    #         # $#$#$#$#$#$ 수정 시작: mask_pred 합치는 방식 선택 (반복문 내)
    #         if seg_mask == None: # inference 시
    #             mask_pred = torch.cat([
    #                 mask_pred[:, :-(M+1), :, :],
    #                 torch.max(mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)[0]
    #             ], dim=1)
    #         elif self.mask_pred_method == 'mean':
    #             mask_pred = torch.cat([
    #                 mask_pred[:, :-(M+1), :, :],
    #                 torch.mean(mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)
    #             ], dim=1)
    #         elif self.mask_pred_method == 'max':
    #             mask_pred = torch.cat([
    #                 mask_pred[:, :-(M+1), :, :],
    #                 torch.max(mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)[0]
    #             ], dim=1)
    #         elif self.mask_pred_method == 'matching':
    #             # mask_pred의 해상도에 맞게 seg_mask를 리사이즈합니다.
    #             target_size = mask_pred.shape[-2:]  # 예: (128, 128)
    #             gt_mask = (seg_mask == 19).float()    # 원래 shape: [B, 1, H, W] (예: 512×512)
                
    #             # learnable 채널들 (마지막 (M+1) 채널)
    #             learnable_channels = mask_pred[:, -(M+1):, :, :]  # shape: [B, M+1, target_H, target_W]
    #             # 조건 없이 gt_mask의 해상도를 예측 결과와 동일하게 맞춥니다.
    #             gt_mask = F.interpolate(gt_mask, size=learnable_channels.shape[-2:], mode='nearest')
    
    #             # 각 채널에 대해 gt_mask 영역의 평균 값을 매칭 스코어로 계산
    #             matching_scores = torch.sum(learnable_channels * gt_mask, dim=(2, 3)) / (torch.sum(gt_mask, dim=(2, 3)) + 1e-6)  # shape: [B, M+1]            
    #             best_channel_idx = matching_scores.argmax(dim=1)  # shape: [B]
                
    #             # 각 배치에서 가장 높은 매칭 스코어를 가진 채널 선택
    #             best_channel = []
    #             for i in range(mask_pred.shape[0]):
    #                 best_channel.append(learnable_channels[i, best_channel_idx[i], :, :].unsqueeze(0))
    #             best_channel = torch.cat(best_channel, dim=0)  # shape: [B, target_H, target_W]
    #             best_channel = best_channel.unsqueeze(1)        # shape: [B, 1, target_H, target_W]
                
    #             # 기존 채널과 선택된 채널을 합침
    #             mask_pred = torch.cat([
    #                 mask_pred[:, :-(M+1), :, :],
    #                 best_channel
    #             ], dim=1)
    #         else:
    #             mask_pred = torch.cat([
    #                 mask_pred[:, :-(M+1), :, :],
    #                 torch.max(mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)
    #             ], dim=1)[0]
    #         # $#$#$#$#$#$ 수정 끝
    
    #         cls_pred_list.append(cls_pred)
    #         mask_pred_list.append(mask_pred)
    
    #     if return_mask_features:
    #         return cls_pred_list, mask_pred_list, mask_features
    #     else:
    #         return cls_pred_list, mask_pred_list


    def forward(self, feats, texts, img_metas, return_mask_features=False, get_similarity=False, seg_mask=None):
        """Forward function.
    
        Args:
            feats (list[Tensor]): Multi scale Features from the upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            seg_mask (Tensor, optional): Semantic segmentation mask. Used for matching method.
        
        Returns:
            tuple: A tuple containing:
                - cls_pred_list (list[Tensor]): Classification logits for each decoder layer.
                  Each is a tensor of shape (batch_size, num_queries, cls_out_channels).
                - mask_pred_list (list[Tensor]): Mask logits for each decoder layer.
                  Each is a tensor of shape (batch_size, num_queries, H, W).
        """
        batch_size = len(img_metas)
        # Pixel decoder (texts를 이용)
        mask_features, multi_scale_memorys = self.pixel_decoder(feats, texts)
        
        # 각 transformer level에 대해 decoder 입력과 positional encoding 계산
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # (B, C, H, W) → (H*W, B, C)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            mask = decoder_input.new_zeros((batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        
        # Query feature 생성: (num_queries, B, C)
        query_feat = self.text_proj(texts).permute(1, 0, 2)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))
        
        cls_pred_list = []
        mask_pred_list = []
        
        # 초기 head 예측
        orig_cls_pred, orig_mask_pred, attn_mask = self.forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:], get_similarity=get_similarity
        )
        
        # M+1: learnable query의 개수 (나머지는 fixed query)
        M = self.M
        
        # -----------------------------
        # Inference (seg_mask is None)와 학습/seg_mask 제공 시 분기 처리
        if seg_mask is None:
            # inference 시
            if self.mask_pred_method == 'matching':
                # fixed 영역: 그대로 사용 (fixed query를 통한 segmask 출력)
                fixed_cls = orig_cls_pred[:, :-(M+1), :]
                fixed_mask = orig_mask_pred[:, :-(M+1), :, :]
                
                # learnable 영역: semantic_inference 스타일로 처리
                learnable_cls = orig_cls_pred[:, -(M+1):, :]    # shape: [B, M+1, num_cls]
                learnable_mask = orig_mask_pred[:, -(M+1):, :, :]  # shape: [B, M+1, H, W]
                # --- query confidence 계산 ---
                # apply softmax on learnable_cls to obtain confidence scores
                learnable_cls_conf = F.softmax(learnable_cls, dim=-1)  # [B, M+1, num_cls]
                # (여기서는 background class(마지막 index)를 제외하여 사용)
                learnable_cls_f = learnable_cls_conf[..., :-1]
                scores, labels = learnable_cls_conf.max(dim=-1)  # scores: [B, M+1], labels: [B, M+1]
                # threshold 조건 (mask2former와 유사)
                keep = (labels.ne(self.num_classes)) & (scores > 0.95) & (labels < 11) & (labels > 1)
                # 각 query의 mask 예측에 confidence를 곱함 (sigmoid된 mask)
                weighted_learnable_mask = learnable_mask.sigmoid() * scores.unsqueeze(-1).unsqueeze(-1)
                # threshold를 만족하지 않는 query는 0으로 처리
                weighted_learnable_mask = weighted_learnable_mask * keep.unsqueeze(-1).unsqueeze(-1).float()
                # learnable 영역의 결과를 aggregate (예: 평균)
                final_learnable_mask = weighted_learnable_mask.mean(dim=1, keepdim=True)
                final_learnable_cls  = learnable_cls.mean(dim=1, keepdim=True)
                
                cls_pred = torch.cat([fixed_cls, final_learnable_cls], dim=1)
                mask_pred = torch.cat([fixed_mask, final_learnable_mask], dim=1)
            elif self.mask_pred_method == 'mean':
                    mask_pred = torch.cat([
                        orig_mask_pred[:, :-(M+1), :, :],
                        torch.mean(orig_mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)
                    ], dim=1)
                    # cls_pred merge: 고정 query와 learnable query 모두 평균 처리
                    cls_pred = torch.cat([
                        orig_cls_pred[:, :-(M+1), :],
                        torch.mean(orig_cls_pred[:, -(M+1):, :], dim=1, keepdim=True)
                    ], dim=1)
                    
            elif self.mask_pred_method == 'max':
                mask_pred = orig_mask_pred
                cls_pred = orig_cls_pred
            elif self.mask_pred_method == 'ood_scoring':
                fixed_cls = orig_cls_pred[:, :-(M+1), :]
                fixed_mask = orig_mask_pred[:, :-(M+1), :, :]
                learnable_cls = orig_cls_pred[:, -(M+1):, :]
                learnable_mask = orig_mask_pred[:, -(M+1):, :, :]

                # query마다 어느 클래스에 들어가는지 확률
                learnable_cls_prob = F.softmax(learnable_cls, dim=-1)

                # 19번째 cls일 확률이 가장 높은 query를 찾기위함
                cls19_prob = learnable_cls_prob[:, :, 19]
                cls19_prob = F.softmax(cls19_prob, dim=1)
                
                final_learnable_mask = (learnable_mask * cls19_prob.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
                final_learnable_cls = (learnable_cls * cls19_prob.unsqueeze(-1)).sum(dim=1, keepdim=True)
                
                cls_pred = torch.cat([fixed_cls, final_learnable_cls], dim=1)
                mask_pred = torch.cat([fixed_mask, final_learnable_mask], dim=1)
            elif self.mask_pred_method == 'ood_scoring_contrastive':
                mask_pred = orig_mask_pred
                cls_pred = orig_cls_pred
            else:
                mask_pred = orig_mask_pred
                cls_pred = orig_cls_pred
        else:
            # Training 또는 seg_mask 제공 시: 기존 방식으로 merge
            if self.mask_pred_method == 'mean':
                mask_pred = torch.cat([
                    orig_mask_pred[:, :-(M+1), :, :],
                    torch.mean(orig_mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)
                ], dim=1)
                # cls_pred merge: 고정 query와 learnable query 모두 평균 처리
                cls_pred = torch.cat([
                    orig_cls_pred[:, :-(M+1), :],
                    torch.mean(orig_cls_pred[:, -(M+1):, :], dim=1, keepdim=True)
                ], dim=1)
            elif self.mask_pred_method == 'max':
                mask_pred = torch.cat([
                    orig_mask_pred[:, :-(M+1), :, :],
                    torch.max(orig_mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)[0]
                ], dim=1)
                # cls_pred merge: 고정 query와 learnable query 모두 평균 처리
                cls_pred = torch.cat([
                    orig_cls_pred[:, :-(M+1), :],
                    torch.mean(orig_cls_pred[:, -(M+1):, :], dim=1, keepdim=True)
                ], dim=1)
            elif self.mask_pred_method == 'matching':
                target_size = orig_mask_pred.shape[-2:]
                gt_mask = (seg_mask == 19).float()
                learnable_channels = orig_mask_pred[:, -(M+1):, :, :]
                gt_mask = F.interpolate(gt_mask, size=learnable_channels.shape[-2:], mode='nearest')
                matching_scores = torch.sum(learnable_channels * gt_mask, dim=(2, 3)) / (torch.sum(gt_mask, dim=(2, 3)) + 1e-6)
                best_channel_idx = matching_scores.argmax(dim=1)
                best_channel = []
                for i in range(orig_mask_pred.shape[0]):
                    best_channel.append(learnable_channels[i, best_channel_idx[i], :, :].unsqueeze(0))
                best_channel = torch.cat(best_channel, dim=0).unsqueeze(1)
                mask_pred = torch.cat([
                    orig_mask_pred[:, :-(M+1), :, :],
                    best_channel
                ], dim=1)
                 # cls_pred merge: 고정 query와 learnable query 모두 평균 처리
                cls_pred = torch.cat([
                    orig_cls_pred[:, :-(M+1), :],
                    torch.mean(orig_cls_pred[:, -(M+1):, :], dim=1, keepdim=True)
                ], dim=1)
            elif self.mask_pred_method == 'ood_scoring':
                fixed_cls = orig_cls_pred[:, :-(M+1), :]
                fixed_mask = orig_mask_pred[:, :-(M+1), :, :]
                learnable_cls = orig_cls_pred[:, -(M+1):, :]
                learnable_mask = orig_mask_pred[:, -(M+1):, :, :]

                # query마다 어느 클래스에 들어가는지 확률
                learnable_cls_prob = F.softmax(learnable_cls, dim=-1)

                # 19번째 cls일 확률이 가장 높은 query를 찾기위함
                cls19_prob = learnable_cls_prob[:, :, 19]
                cls19_prob = F.softmax(cls19_prob, dim=1)
                
                final_learnable_mask = (learnable_mask * cls19_prob.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
                final_learnable_cls = (learnable_cls * cls19_prob.unsqueeze(-1)).sum(dim=1, keepdim=True)
                
                cls_pred = torch.cat([fixed_cls, final_learnable_cls], dim=1)
                mask_pred = torch.cat([fixed_mask, final_learnable_mask], dim=1) 
            elif self.mask_pred_method == 'ood_scoring_contrastive':
                fixed_cls = orig_cls_pred[:, :-(M+1), :]
                fixed_mask = orig_mask_pred[:, :-(M+1), :, :]
                learnable_cls = orig_cls_pred[:, -(M+1):, :]
                learnable_mask = orig_mask_pred[:, -(M+1):, :, :]
            
                # query마다 어느 클래스에 들어가는지 확률
                learnable_cls_prob = F.softmax(learnable_cls, dim=-1)
            
                # 19번째 cls일 확률이 가장 높은 query를 찾기위함
                cls19_prob = learnable_cls_prob[:, :, 19]
                
                # ood contrastive 가중치 계산
                ood_contrastive_weights = F.softmax(cls19_prob, dim=1)
            
                # 19번째 cls일 확률이 가장 높은 query의 index 찾기
                max_cls19_idx = torch.argmax(cls19_prob, dim=1, keepdim=True)
            
                # 그에 해당하는 mask와 cls 선택
                final_learnable_mask = torch.gather(learnable_mask, 1, max_cls19_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, learnable_mask.shape[2], learnable_mask.shape[3]))
                final_learnable_cls = torch.gather(learnable_cls, 1, max_cls19_idx.unsqueeze(-1).repeat(1,1,learnable_cls.shape[2]))
            
                # 선택된 cls에 ood contrastive 가중치 곱하기
                # 수정된 부분: 해당 query에 해당하는 가중치만 곱함
                # print("final_learnable_cls.shape",final_learnable_cls.shape)
                # print("ood_contrastive_weights.shape",ood_contrastive_weights.shape)
                # print("torch.gather(ood_contrastive_weights, 1, max_cls19_idx)",torch.gather(ood_contrastive_weights, 1, max_cls19_idx))
                final_learnable_cls = final_learnable_cls * torch.gather(ood_contrastive_weights, 1, max_cls19_idx) .unsqueeze(-1)
            
                # cls contrastive 가중치 계산
                cls_contrastive_weights = F.softmax(orig_cls_pred, dim=-1)
                
                # fixed cls에 대한 가중치 적용 (19번째 클래스와 멀어지도록)
                fixed_cls_weights = cls_contrastive_weights[:, :-(M+1), :]
                
                # background로 예측하도록 남은 fixed cls들의 평균을 구함.
                # remaining_fixed_cls = fixed_cls * (1-fixed_cls_weights[:,:,19].unsqueeze(-1))
                # background_cls = remaining_fixed_cls.mean(dim=1, keepdim=True) #  [B, 1, C]
                
                # fixed_cls에서 19번째 cls에 해당하는 값은 줄여주고 나머지는 그대로 둔다
                # 존재하는 cls들과 멀어지도록 유도.
                fixed_cls = fixed_cls * (1 - fixed_cls_weights[:,:,19].unsqueeze(-1)) 
                
                # 최종 cls_pred 생성: [:, :19, :] + [:, 20, :] + [:, 21, :]
                # cls_pred = torch.cat([fixed_cls, final_learnable_cls, background_cls], dim=1)
                cls_pred = torch.cat([fixed_cls, final_learnable_cls], dim=1)
                mask_pred = torch.cat([fixed_mask, final_learnable_mask], dim=1)
                # print("cls_pred.shape",cls_pred.shape)
                # print("mask_pred.shape",mask_pred.shape)
            else:
                mask_pred = orig_mask_pred
                cls_pred = orig_cls_pred


        
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)
        
        # -----------------------------
        # Transformer decoder layers
        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # 만약 attn_mask의 모든 값이 True이면 모두 False로 변경 (background 처리)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            
            layer = self.transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                key_padding_mask=None
            )
            # 새로운 head 예측 (각 decoder layer마다 업데이트)
            orig_cls_pred, orig_mask_pred, attn_mask = self.forward_head(
                query_feat, mask_features,
                multi_scale_memorys[(i+1) % self.num_transformer_feat_level].shape[-2:], get_similarity=get_similarity
            )
            
            if seg_mask is None:
                if self.mask_pred_method == 'matching':
                    fixed_cls = orig_cls_pred[:, :-(M+1), :]
                    fixed_mask = orig_mask_pred[:, :-(M+1), :, :]
                    learnable_cls = orig_cls_pred[:, -(M+1):, :]
                    learnable_mask = orig_mask_pred[:, -(M+1):, :, :]
                    
                    learnable_cls_conf = F.softmax(learnable_cls, dim=-1)
                    learnable_cls_f = learnable_cls_conf[..., :-1]
                    scores, labels = learnable_cls_conf.max(dim=-1)
                    keep = (labels.ne(self.num_classes)) & (scores > 0.95) & (labels < 11) & (labels > 1)
                    weighted_learnable_mask = learnable_mask.sigmoid() * scores.unsqueeze(-1).unsqueeze(-1)
                    weighted_learnable_mask = weighted_learnable_mask * keep.unsqueeze(-1).unsqueeze(-1).float()
                    final_learnable_mask = weighted_learnable_mask.mean(dim=1, keepdim=True)
                    final_learnable_cls  = learnable_cls.mean(dim=1, keepdim=True)
                    
                    cls_pred = torch.cat([fixed_cls, final_learnable_cls], dim=1)
                    mask_pred = torch.cat([fixed_mask, final_learnable_mask], dim=1)
                else:

                    if self.mask_pred_method == 'mean':
                        mask_pred = torch.cat([
                            orig_mask_pred[:, :-(M+1), :, :],
                            torch.mean(orig_mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)
                        ], dim=1)
                        cls_pred = torch.cat([
                            orig_cls_pred[:, :-(M+1), :],
                            torch.mean(orig_cls_pred[:, -(M+1):, :], dim=1, keepdim=True)
                        ], dim=1)
                    elif self.mask_pred_method == 'max':
                        mask_pred = orig_mask_pred
                        cls_pred = orig_cls_pred
                    elif self.mask_pred_method == 'ood_scoring':
                        fixed_cls = orig_cls_pred[:, :-(M+1), :]
                        fixed_mask = orig_mask_pred[:, :-(M+1), :, :]
                        learnable_cls = orig_cls_pred[:, -(M+1):, :]
                        learnable_mask = orig_mask_pred[:, -(M+1):, :, :]
        
                        # query마다 어느 클래스에 들어가는지 확률
                        learnable_cls_prob = F.softmax(learnable_cls, dim=-1)
        
                        # 19번째 cls일 확률이 가장 높은 query를 찾기위함
                        cls19_prob = learnable_cls_prob[:, :, 19]
                        cls19_prob = F.softmax(cls19_prob, dim=1)
                        
                        final_learnable_mask = (learnable_mask * cls19_prob.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
                        final_learnable_cls = (learnable_cls * cls19_prob.unsqueeze(-1)).sum(dim=1, keepdim=True)
                        
                        cls_pred = torch.cat([fixed_cls, final_learnable_cls], dim=1)
                        mask_pred = torch.cat([fixed_mask, final_learnable_mask], dim=1) 
                    elif self.mask_pred_method == 'ood_scoring_contrastive':
                        mask_pred = orig_mask_pred
                        cls_pred = orig_cls_pred
                    else:
                        mask_pred = orig_mask_pred
                        cls_pred = orig_cls_pred
            else:
                if self.mask_pred_method == 'mean':
                    mask_pred = torch.cat([
                        orig_mask_pred[:, :-(M+1), :, :],
                        torch.mean(orig_mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)
                    ], dim=1)
                    cls_pred = torch.cat([
                        orig_cls_pred[:, :-(M+1), :],
                        torch.mean(orig_cls_pred[:, -(M+1):, :], dim=1, keepdim=True)
                    ], dim=1)
                elif self.mask_pred_method == 'max':
                    mask_pred = torch.cat([
                        orig_mask_pred[:, :-(M+1), :, :],
                        torch.max(orig_mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)[0]
                    ], dim=1)
                    cls_pred = torch.cat([
                        orig_cls_pred[:, :-(M+1), :],
                        torch.mean(orig_cls_pred[:, -(M+1):, :], dim=1, keepdim=True)
                    ], dim=1)
                elif self.mask_pred_method == 'matching':
                    target_size = orig_mask_pred.shape[-2:]
                    gt_mask = (seg_mask == 19).float()
                    learnable_channels = orig_mask_pred[:, -(M+1):, :, :]
                    gt_mask = F.interpolate(gt_mask, size=learnable_channels.shape[-2:], mode='nearest')
                    matching_scores = torch.sum(learnable_channels * gt_mask, dim=(2, 3)) / (torch.sum(gt_mask, dim=(2, 3)) + 1e-6)
                    best_channel_idx = matching_scores.argmax(dim=1)
                    best_channel = []
                    for j in range(orig_mask_pred.shape[0]):
                        best_channel.append(learnable_channels[j, best_channel_idx[j], :, :].unsqueeze(0))
                    best_channel = torch.cat(best_channel, dim=0).unsqueeze(1)
                    mask_pred = torch.cat([
                        orig_mask_pred[:, :-(M+1), :, :],
                        best_channel
                    ], dim=1)
                    cls_pred = torch.cat([
                        orig_cls_pred[:, :-(M+1), :],
                        torch.mean(orig_cls_pred[:, -(M+1):, :], dim=1, keepdim=True)
                    ], dim=1)
                elif self.mask_pred_method == 'ood_scoring':
                    fixed_cls = orig_cls_pred[:, :-(M+1), :]
                    fixed_mask = orig_mask_pred[:, :-(M+1), :, :]
                    learnable_cls = orig_cls_pred[:, -(M+1):, :]
                    learnable_mask = orig_mask_pred[:, -(M+1):, :, :]
    
                    # query마다 어느 클래스에 들어가는지 확률
                    learnable_cls_prob = F.softmax(learnable_cls, dim=-1)
    
                    # 19번째 cls일 확률이 가장 높은 query를 찾기위함
                    cls19_prob = learnable_cls_prob[:, :, 19]
                    cls19_prob = F.softmax(cls19_prob, dim=1)
                    
                    final_learnable_mask = (learnable_mask * cls19_prob.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
                    final_learnable_cls = (learnable_cls * cls19_prob.unsqueeze(-1)).sum(dim=1, keepdim=True)
                    
                    cls_pred = torch.cat([fixed_cls, final_learnable_cls], dim=1)
                    mask_pred = torch.cat([fixed_mask, final_learnable_mask], dim=1) 
                elif self.mask_pred_method == 'ood_scoring_contrastive':
                    
                    fixed_cls = orig_cls_pred[:, :-(M+1), :]
                    fixed_mask = orig_mask_pred[:, :-(M+1), :, :]
                    learnable_cls = orig_cls_pred[:, -(M+1):, :]
                    learnable_mask = orig_mask_pred[:, -(M+1):, :, :]

                
                    # query마다 어느 클래스에 들어가는지 확률
                    learnable_cls_prob = F.softmax(learnable_cls, dim=-1)
                
                    # 19번째 cls일 확률이 가장 높은 query를 찾기위함
                    cls19_prob = learnable_cls_prob[:, :, 19]
                    
                    # ood contrastive 가중치 계산
                    ood_contrastive_weights = F.softmax(cls19_prob, dim=1)
                
                    # 19번째 cls일 확률이 가장 높은 query의 index 찾기
                    max_cls19_idx = torch.argmax(cls19_prob, dim=1, keepdim=True)
                
                    # 그에 해당하는 mask와 cls 선택
                    final_learnable_mask = torch.gather(learnable_mask, 1, max_cls19_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, learnable_mask.shape[2], learnable_mask.shape[3]))
                    final_learnable_cls = torch.gather(learnable_cls, 1, max_cls19_idx.unsqueeze(-1).repeat(1,1,learnable_cls.shape[2]))

                    # 선택된 cls에 ood contrastive 가중치 곱하기
                    # 수정된 부분: 해당 query에 해당하는 가중치만 곱함
                    final_learnable_cls = final_learnable_cls * torch.gather(ood_contrastive_weights, 1, max_cls19_idx) .unsqueeze(-1)
                
                    # cls contrastive 가중치 계산
                    cls_contrastive_weights = F.softmax(orig_cls_pred, dim=-1)
                    
                    # fixed cls에 대한 가중치 적용 (19번째 클래스와 멀어지도록)
                    fixed_cls_weights = cls_contrastive_weights[:, :-(M+1), :]
                    
                    # background로 예측하도록 남은 fixed cls들의 평균을 구함.
                    remaining_fixed_cls = fixed_cls * (1-fixed_cls_weights[:,:,19].unsqueeze(-1))
                    background_cls = remaining_fixed_cls.mean(dim=1, keepdim=True) #  [B, 1, C]
                    
                    # fixed_cls에서 19번째 cls에 해당하는 값은 줄여주고 나머지는 그대로 둔다
                    fixed_cls = fixed_cls * (1 - fixed_cls_weights[:,:,19].unsqueeze(-1)) 
                    
                    # 최종 cls_pred 생성: [:, :19, :] + [:, 20, :] + [:, 21, :]
                    # cls_pred = torch.cat([fixed_cls, final_learnable_cls, background_cls], dim=1)
                    cls_pred = torch.cat([fixed_cls, final_learnable_cls], dim=1)
                    mask_pred = torch.cat([fixed_mask, final_learnable_mask], dim=1)

                    # print("cls_pred.shape",cls_pred.shape)
                    # print("mask_pred.shape",mask_pred.shape)
                else:
                    mask_pred = orig_mask_pred
                    cls_pred = orig_cls_pred
                

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
        
        if return_mask_features:
            return cls_pred_list, mask_pred_list, mask_features
        else:
            return cls_pred_list, mask_pred_list
    

    
    def forward_train(self, x, texts, img_metas, gt_semantic_seg, train_cfg,
                      gt_labels, gt_masks):
        # print("len(gt_labels)",len(gt_labels))
        # print("gt_labels[0].shape",gt_labels[0].shape)
        # print("gt_labels[0].shape",gt_labels[1].shape)
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Multi-level features from the upstream network,
                each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_semantic_seg (list[tensor]):Each element is the ground truth
                of semantic segmentation with the shape (N, H, W).
            train_cfg (dict): The training config, which not been used in
                maskformer.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, h, w).

        Returns:
            losses (dict[str, Tensor]): a dictionary of loss components
        """

        # forward
        # original
        # all_cls_scores, all_mask_preds = self(x, texts, img_metas)

        all_cls_scores, all_mask_preds = self(x, texts, img_metas, seg_mask = gt_semantic_seg)

        # loss
        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks,
                           img_metas)

        return losses

    # def forward_test(self, inputs, texts, img_metas, test_cfg):
    #     """Test segment without test-time aumengtation.

    #     Only the output of last decoder layers was used.

    #     Args:
    #         inputs (list[Tensor]): Multi-level features from the
    #             upstream network, each is a 4D-tensor.
    #         img_metas (list[dict]): List of image information.
    #         test_cfg (dict): Testing config.

    #     Returns:
    #         seg_mask (Tensor): Predicted semantic segmentation logits.
    #     """
    #     all_cls_scores, all_mask_preds = self(inputs, texts, img_metas)
    #     cls_score, mask_pred = all_cls_scores[-1], all_mask_preds[-1]
    #     ori_h, ori_w, _ = img_metas[0]['ori_shape']

    #     # semantic inference
    #     cls_score = F.softmax(cls_score, dim=-1)[..., :-1]
    #     mask_pred = mask_pred.sigmoid()
        
    #     seg_mask = torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred)
        
    #     # print("cls_score.shape",cls_score.shape)
    #     # print("mask_pred.shape",mask_pred.shape)
    #     # print("seg_mask.shape",seg_mask.shape)
    #     # cls_score.shape torch.Size([1, 20, 20])
    #     # mask_pred.shape torch.Size([1, 20, 128, 128])
    #     # seg_mask.shape torch.Size([1, 20, 128, 128])
    #     return seg_mask
    def forward_test(self, inputs, texts, img_metas, test_cfg):
        """Test segment without test-time aumengtation.

        Only the output of last decoder layers was used.

        Args:
            inputs (list[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            test_cfg (dict): Testing config.

        Returns:
            seg_mask (Tensor): Predicted semantic segmentation logits.
        """
        all_cls_scores, all_mask_preds = self(inputs, texts, img_metas)
        cls_score, mask_pred = all_cls_scores[-1], all_mask_preds[-1]
        ori_h, ori_w, _ = img_metas[0]['ori_shape']

        if self.mask_pred_method == 'max':
            # ver_1
            M = self.M
            mask_pred = torch.cat([
                mask_pred[:, :-(M+1), :, :],
                torch.max(mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)[0]
            ], dim=1)
            cls_score = torch.cat([
                cls_score[:, :-(M+1), :],
                torch.mean(cls_score[:, -(M+1):, :], dim=1, keepdim=True)
            ], dim=1)
            cls_score = F.softmax(cls_score, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            
            seg_mask = torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred)

            # # ver_2
            # M = self.M
            
            # # ID와 OOD 부분 분리
            # id_cls_score = cls_score[:, :-(M+1), :]
            # id_mask_pred = mask_pred[:, :-(M+1), :, :]
            # ood_cls_score = cls_score[:, -(M+1):, :]
            # ood_mask_pred = mask_pred[:, -(M+1):, :, :]
            
            # # 각 배치에서 19번 OOD class의 확률이 가장 높은 query 찾기
            # ood_cls_prob = F.softmax(ood_cls_score, dim=-1)[:, :, 19]  # (B, M+1)
            # max_ood_idx = torch.argmax(ood_cls_prob, dim=1)  # (B,)
            
            # # 해당 query의 mask 및 cls_score만 선택
            # selected_ood_mask = torch.stack([
            #     ood_mask_pred[i, max_ood_idx[i], :, :]
            #     for i in range(ood_mask_pred.shape[0])
            # ], dim=0).unsqueeze(1)  # (B, 1, H, W)
            
            # selected_ood_cls = torch.stack([
            #     ood_cls_score[i, max_ood_idx[i], :]
            #     for i in range(ood_cls_score.shape[0])
            # ], dim=0).unsqueeze(1)  # (B, 1, num_classes)
            
            # # ID 부분과 결합
            # mask_pred = torch.cat([id_mask_pred, selected_ood_mask], dim=1)
            # cls_score = torch.cat([id_cls_score, selected_ood_cls], dim=1)
            
            # # Semantic inference
            # cls_score = F.softmax(cls_score, dim=-1)[..., :-1]
            # mask_pred = mask_pred.sigmoid()

            # seg_mask = torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred)
          
        elif self.mask_pred_method == 'ood_scoring_contrastive':
            M = self.M
            cls_score = F.softmax(cls_score, dim=-1)[..., :-1] # cls_score.shape torch.Size([1, 70, 20])
            mask_pred = mask_pred.sigmoid() # mask_pred.shape torch.Size([1, 70, 128, 128])

            print("cls_score.shape",cls_score.shape)
            print("mask_pred.shape",mask_pred.shape)
            id_cls_score = cls_score[:, :-(M+1), :]
            id_mask_pred = mask_pred[:, :-(M+1), :, :]
            ood_cls_score = cls_score[:, -(M+1):, :]
            ood_mask_pred = mask_pred[:, -(M+1):, :, :]
            print("id_cls_score.shape",id_cls_score.shape) # id_cls_score.shape torch.Size([1, 19, 20])
            print("id_mask_pred.shape",id_mask_pred.shape) # id_mask_pred.shape torch.Size([1, 19, 128, 128
            
            print("ood_cls_score.shape",ood_cls_score.shape) # ood_cls_score.shape torch.Size([1, 51, 20])
            print("ood_mask_pred.shape",ood_mask_pred.shape) # ood_mask_pred.shape torch.Size([1, 51, 128, 128])
            if True:
                # query마다 어느 클래스에 들어가는지 확률
                ood_cls_score_prob = F.softmax(ood_cls_score, dim=-1)

                # ood contrastive 가중치 계산
                # ood_contrastive_weights = F.softmax(ood_cls_score_prob, dim=1)
                                
                ood_cls_score = ood_cls_score * ood_cls_score_prob[:, :, 19].unsqueeze(-1)
            
            # if True:
            #     # cls contrastive 가중치 계산
            #     cls_contrastive_weights = F.softmax(cls_score/0.01, dim=-1)
            #     # fixed cls에 대한 가중치 적용 (19번째 클래스와 멀어지도록)
            #     fixed_cls_weights = cls_contrastive_weights[:, :-(M+1), :]
            #     # background로 예측하도록 남은 fixed cls들의 평균을 구함.
            #     id_cls_score = id_cls_score * (1-fixed_cls_weights[:,:,19].unsqueeze(-1))
                
            
            id_seg_mask = torch.einsum('bqc,bqhw->bchw', id_cls_score, id_mask_pred)
            ood_seg_mask = torch.einsum('bqc,bqhw->bchw', ood_cls_score, ood_mask_pred)
            
            print("id_seg_mask.shape",id_seg_mask.shape) # id_seg_mask.shape torch.Size([1, 20, 128, 128])
            print("ood_seg_mask.shape",ood_seg_mask.shape) # ood_seg_mask.shape torch.Size([1, 20, 128, 128])

            seg_mask = torch.cat([id_seg_mask[:,:19,:,:], ood_seg_mask[:,19:,:,:]], dim=1)
            
        else:
            # semantic inference
            cls_score = F.softmax(cls_score, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            
            seg_mask = torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred)
            
            # print("cls_score.shape",cls_score.shape)
            # print("mask_pred.shape",mask_pred.shape)
            # print("seg_mask.shape",seg_mask.shape)
            # cls_score.shape torch.Size([1, 20, 20])
            # mask_pred.shape torch.Size([1, 20, 128, 128])
            # seg_mask.shape torch.Size([1, 20, 128, 128])
        return seg_mask

    def forward_inference(self, inputs, texts, img_metas, test_cfg):
        all_cls_scores, all_mask_preds, mask_features =\
            self(inputs, texts, img_metas, return_mask_features=True)
        return all_mask_preds, mask_features

###############################################################################################################

# import copy
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.cnn import Conv2d, build_plugin_layer, caffe2_xavier_init, ConvModule
# from mmcv.cnn.bricks.transformer import (build_positional_encoding,
#                                          build_transformer_layer_sequence)
# from mmcv.ops import point_sample
# from mmcv.runner import ModuleList, force_fp32
# from mmseg.models.builder import HEADS, build_loss
# from mmseg.models.decode_heads.decode_head import BaseDecodeHead
# from ...core import build_sampler, multi_apply, reduce_mean
# from ..builder import build_assigner
# from ..utils import get_uncertain_point_coords_with_randomness
# # 필요시 Hungarian matching을 위한 라이브러리 임포트 (예: scipy)
# from scipy.optimize import linear_sum_assignment  # NEW CODE

# @HEADS.register_module()
# class tqdmHead(BaseDecodeHead):

#     def __init__(self,
#                  in_channels,
#                  feat_channels,
#                  out_channels,
#                  num_things_classes=80,
#                  num_stuff_classes=53,
#                  num_queries=100,
#                  num_transformer_feat_level=3,
#                  pixel_decoder=None,
#                  enforce_decoder_input_project=False,
#                  transformer_decoder=None,
#                  positional_encoding=None,
#                  loss_cls=None,
#                  loss_mask=None,
#                  loss_dice=None,
#                  train_cfg=None,
#                  test_cfg=None,
#                  init_cfg=None,
#                  text_proj=None,
#                  **kwargs):
#         super(tqdmHead, self).__init__(
#             in_channels=in_channels,
#             channels=feat_channels,
#             num_classes=(num_things_classes + num_stuff_classes),
#             init_cfg=init_cfg,
#             input_transform='multiple_select',
#             **kwargs)
#         self.num_things_classes = num_things_classes
#         self.num_stuff_classes = num_stuff_classes
#         self.num_classes = self.num_things_classes + self.num_stuff_classes
#         self.num_queries = num_queries
#         self.num_transformer_feat_level = num_transformer_feat_level
#         self.num_heads = transformer_decoder.transformerlayers.attn_cfgs.num_heads
#         self.num_transformer_decoder_layers = transformer_decoder.num_layers
#         assert pixel_decoder.encoder.transformerlayers.attn_cfgs[0].num_levels == num_transformer_feat_level
#         pixel_decoder_ = copy.deepcopy(pixel_decoder)
#         pixel_decoder_.update(
#             in_channels=in_channels,
#             feat_channels=feat_channels,
#             out_channels=out_channels,
#             text_proj=text_proj)
#         self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]
#         self.transformer_decoder = build_transformer_layer_sequence(transformer_decoder)
#         self.decoder_embed_dims = self.transformer_decoder.embed_dims

#         self.decoder_input_projs = ModuleList()
#         # from low resolution to high resolution
#         for _ in range(num_transformer_feat_level):
#             if (self.decoder_embed_dims != feat_channels or enforce_decoder_input_project):
#                 self.decoder_input_projs.append(
#                     Conv2d(feat_channels, self.decoder_embed_dims, kernel_size=1))
#             else:
#                 self.decoder_input_projs.append(nn.Identity())
#         self.decoder_positional_encoding = build_positional_encoding(positional_encoding)
#         self.query_embed = nn.Embedding(self.num_queries, feat_channels)
#         self.text_proj = nn.Sequential(
#             nn.Linear(text_proj.text_in_dim, text_proj.text_out_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(text_proj.text_out_dim, text_proj.text_out_dim))
#         # from low resolution to high resolution
#         self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)

#         self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
#         self.mask_embed = nn.Sequential(
#             nn.Linear(feat_channels, feat_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(feat_channels, feat_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(feat_channels, out_channels))
#         self.conv_seg = None  # fix a bug here (conv_seg is not used)

#         self.test_cfg = test_cfg
#         self.train_cfg = train_cfg
#         if train_cfg:
#             self.assigner = build_assigner(self.train_cfg.assigner)
#             self.sampler = build_sampler(self.train_cfg.sampler, context=self)
#             self.num_points = self.train_cfg.get('num_points', 12544)
#             self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
#             self.importance_sample_ratio = self.train_cfg.get('importance_sample_ratio', 0.75)

#         self.class_weight = loss_cls.class_weight
#         self.loss_cls = build_loss(loss_cls)
#         self.loss_mask = build_loss(loss_mask)
#         self.loss_dice = build_loss(loss_dice)

#         ##### #$#$#$#$#$ method 4
#         # self.loss_contrastive = build_loss({'type': 'MaskContrastiveLoss'})
#         # self.ood_score_conv = nn.Conv2d(in_channels=20, out_channels=1, kernel_size=1)
#         ##### #$#$#$#$#$#

#         ##### #$#$#$#$#$#
#         # self.mask_pred_method = 'ood_scoring_contrastive'# 'matching'
#         self.mask_pred_method = 'max'
#         self.M = self.num_queries - self.num_classes
#         ##### #$#$#$#$#$#

#         ### NEW CODE for GPT bipartite contrastive
#         # 새로 추가되는 코드는 self.gpt_bipartite_contrastive 플래그가 True일 때 동작함.
#         self.gpt_bipartite_contrastive = kwargs.get('gpt_bipartite_contrastive', False)
#         if self.gpt_bipartite_contrastive:
#             # repulsion loss의 계수를 설정 (하이퍼파라미터)
#             self.lambda_repulsion = kwargs.get('lambda_repulsion', 1.0)

#     def init_weights(self):
#         for m in self.decoder_input_projs:
#             if isinstance(m, Conv2d):
#                 caffe2_xavier_init(m, bias=0)
#         self.pixel_decoder.init_weights()
#         for p in self.transformer_decoder.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_normal_(p)

#     def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
#                     gt_masks_list, img_metas):
#         (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
#          pos_inds_list, neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list,
#                                                      mask_preds_list, gt_labels_list,
#                                                      gt_masks_list, img_metas)
#         num_total_pos = sum((inds.numel() for inds in pos_inds_list))
#         num_total_neg = sum((inds.numel() for inds in neg_inds_list))
#         return (labels_list, label_weights_list, mask_targets_list,
#                 mask_weights_list, num_total_pos, num_total_neg)

#     def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks, img_metas):
#         # sample points
#         num_queries = cls_score.shape[0]
#         num_gts = gt_labels.shape[0]
#         point_coords = torch.rand((1, self.num_points, 2), device=cls_score.device)
#         mask_points_pred = point_sample(mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1, 1)).squeeze(1)
#         gt_points_masks = point_sample(gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1, 1)).squeeze(1)

#         # assign and sample (기존 assigner를 사용)
#         assign_result = self.assigner.assign(cls_score, mask_points_pred, gt_labels, gt_points_masks, img_metas)
#         # NEW CODE: OOD bipartite matching 적용 (플래그 활성시)
#         if self.gpt_bipartite_contrastive:
#             # 예를 들어, fixed query는 0 ~ (self.num_classes - 1)이고, OOD query는 그 이후라고 가정합니다.
#             # ood_indices = list(range(self.num_classes, num_queries))
#             # 여기서 GT와 OOD query 간 비용 행렬(cost_matrix)을 계산한 후 Hungarian matching을 수행합니다.
#             # 아래는 placeholder 코드입니다.
#             # cost_matrix = compute_cost(ood_query_embeds, gt_masks)  # 사용자가 구현해야 함
#             # row_ind, col_ind = linear_sum_assignment(cost_matrix)
#             # assign_result.ood_assignment = (row_ind, col_ind)
#             pass

#         sampling_result = self.sampler.sample(assign_result, mask_pred, gt_masks)
#         pos_inds = sampling_result.pos_inds
#         neg_inds = sampling_result.neg_inds

#         # label target
#         labels = gt_labels.new_full((self.num_classes,), self.num_classes, dtype=torch.long)
#         labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
#         label_weights = gt_labels.new_ones((self.num_classes,))
#         mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
#         mask_weights = mask_pred.new_zeros((self.num_classes,))
#         mask_weights[pos_inds] = 1.0

#         return (labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds)

#     def loss_single(self, cls_scores, mask_preds, gt_labels_list, gt_masks_list, img_metas):
#         num_imgs = cls_scores.size(0)
#         cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
#         mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
#         (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
#          num_total_pos, num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list,
#                                                           gt_labels_list, gt_masks_list, img_metas)
#         labels = torch.stack(labels_list, dim=0)
#         label_weights = torch.stack(label_weights_list, dim=0)
#         mask_targets = torch.cat(mask_targets_list, dim=0)
#         mask_weights = torch.stack(mask_weights_list, dim=0)
#         cls_scores = cls_scores.flatten(0, 1)
#         labels = labels.flatten(0, 1)
#         label_weights = label_weights.flatten(0, 1)
#         class_weight = cls_scores.new_tensor(self.class_weight)
#         loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=class_weight[labels].sum())

#         num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
#         num_total_masks = max(num_total_masks, 1)
#         mask_preds = mask_preds[mask_weights > 0]
#         if mask_targets.shape[0] == 0:
#             loss_dice = mask_preds.sum()
#             loss_mask = mask_preds.sum()
#             base_loss = (loss_cls, loss_mask, loss_dice)
#         else:
#             with torch.no_grad():
#                 points_coords = get_uncertain_point_coords_with_randomness(mask_preds.unsqueeze(1), None,
#                                                                            self.num_points,
#                                                                            self.oversample_ratio,
#                                                                            self.importance_sample_ratio)
#                 mask_point_targets = point_sample(mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
#             mask_point_preds = point_sample(mask_preds.unsqueeze(1), points_coords).squeeze(1)
#             loss_dice = self.loss_dice(mask_point_preds, mask_point_targets, avg_factor=num_total_masks)
#             mask_point_preds = mask_point_preds.reshape(-1, 1)
#             mask_point_targets = mask_point_targets.reshape(-1)
#             loss_mask = self.loss_mask(mask_point_preds, mask_point_targets, avg_factor=num_total_masks * self.num_points)
#             base_loss = (loss_cls, loss_mask, loss_dice)

#         return base_loss

#     @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
#     def loss(self, all_cls_scores, all_mask_preds, gt_labels_list, gt_masks_list, img_metas, ood_query_embeds=None):
#         num_dec_layers = len(all_cls_scores)
#         all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
#         all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
#         img_metas_list = [img_metas for _ in range(num_dec_layers)]
#         losses_cls, losses_mask, losses_dice = multi_apply(self.loss_single, all_cls_scores, all_mask_preds,
#                                                             all_gt_labels_list, all_gt_masks_list, img_metas_list)
#         loss_dict = dict()
#         loss_dict['loss_cls'] = losses_cls[-1]
#         loss_dict['loss_mask'] = losses_mask[-1]
#         loss_dict['loss_dice'] = losses_dice[-1]
#         num_dec_layer = 0
#         for loss_cls_i, loss_mask_i, loss_dice_i in zip(losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
#             loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
#             loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
#             loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
#             num_dec_layer += 1

#         ### NEW CODE: OOD Query repulsion loss 추가 (Query Collapse 방지)
#         if self.gpt_bipartite_contrastive and ood_query_embeds is not None:
#             repulsion_loss = self.compute_repulsion_loss(ood_query_embeds)
#             loss_dict['loss_repulsion'] = repulsion_loss * self.lambda_repulsion

#         return loss_dict

#     def compute_repulsion_loss(self, ood_query_embeds):
#         """
#         OOD query embedding들 사이의 pairwise cosine similarity가 너무 높으면 패널티를 부여합니다.
#         여기서는 대각선을 제외한 모든 쌍에 대해 (cos_sim - tau)를 clamped하여 평균을 구하는 예시입니다.
#         """
#         # ood_query_embeds: (N, C)
#         N = ood_query_embeds.shape[0]
#         if N < 2:
#             return torch.tensor(0.0, device=ood_query_embeds.device)
#         norm_embeds = F.normalize(ood_query_embeds, dim=-1)
#         sim_matrix = torch.matmul(norm_embeds, norm_embeds.t())  # (N, N)
#         diag_mask = torch.eye(N, device=ood_query_embeds.device).bool()
#         sim_matrix = sim_matrix.masked_fill(diag_mask, 0)
#         # threshold tau = 0.5 (예시)
#         tau = 0.5
#         loss = torch.clamp(sim_matrix - tau, min=0).mean()
#         return loss

#     def forward_head(self, decoder_out, mask_feature, attn_mask_target_size, get_similarity=False):
#         decoder_out = self.transformer_decoder.post_norm(decoder_out)
#         decoder_out = decoder_out.transpose(0, 1)
#         cls_pred = self.cls_embed(decoder_out)
#         mask_embed = self.mask_embed(decoder_out)
#         mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
#         attn_mask = F.interpolate(mask_pred, attn_mask_target_size, mode='bilinear', align_corners=False)
#         attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat((1, self.num_heads, 1, 1)).flatten(0, 1)
#         attn_mask = attn_mask.sigmoid() < 0.5
#         attn_mask = attn_mask.detach()
#         if not get_similarity:
#             return cls_pred, mask_pred, attn_mask
#         else:
#             mask_feature = F.normalize(mask_feature, dim=1, p=2)
#             mask_embed = F.normalize(mask_embed, dim=-1, p=2)
#             sim = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
#             sim = (sim + 1.) / 2.
#             return cls_pred, sim, attn_mask

#     def forward(self, feats, texts, img_metas, return_mask_features=False, get_similarity=False, seg_mask=None):
#         batch_size = len(img_metas)
#         mask_features, multi_scale_memorys = self.pixel_decoder(feats, texts)
#         decoder_inputs = []
#         decoder_positional_encodings = []
#         for i in range(self.num_transformer_feat_level):
#             decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
#             decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
#             level_embed = self.level_embed.weight[i].view(1, 1, -1)
#             decoder_input = decoder_input + level_embed
#             mask = decoder_input.new_zeros((batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool)
#             decoder_positional_encoding = self.decoder_positional_encoding(mask)
#             decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(2, 0, 1)
#             decoder_inputs.append(decoder_input)
#             decoder_positional_encodings.append(decoder_positional_encoding)
#         query_feat = self.text_proj(texts).permute(1, 0, 2)
#         query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))
        
#         ### NEW CODE: OOD query embedding 추출
#         if self.gpt_bipartite_contrastive:
#             # fixed query: 0 ~ (self.num_classes - 1)
#             # OOD query: self.num_classes ~ (self.num_queries - 1)
#             ood_query_embeds = query_feat[self.num_classes:self.num_queries]
#         else:
#             ood_query_embeds = None

#         cls_pred_list = []
#         mask_pred_list = []
#         orig_cls_pred, orig_mask_pred, attn_mask = self.forward_head(query_feat, mask_features, multi_scale_memorys[0].shape[-2:], get_similarity=get_similarity)
#         M = self.M

#         if seg_mask is None:
#             if self.mask_pred_method == 'max':
#                 mask_pred = orig_mask_pred
#                 cls_pred = orig_cls_pred

#         else:
#             if self.mask_pred_method == 'max':
#                 mask_pred = torch.cat([
#                     orig_mask_pred[:, :-(M+1), :, :],
#                     torch.max(orig_mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)[0]
#                 ], dim=1)
#                 cls_pred = torch.cat([
#                     orig_cls_pred[:, :-(M+1), :],
#                     torch.mean(orig_cls_pred[:, -(M+1):, :], dim=1, keepdim=True)
#                 ], dim=1)

#             else:
#                 mask_pred = orig_mask_pred
#                 cls_pred = orig_cls_pred

#         cls_pred_list.append(cls_pred)
#         mask_pred_list.append(mask_pred)
        
#         for i in range(self.num_transformer_decoder_layers):
#             level_idx = i % self.num_transformer_feat_level
#             attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
#             layer = self.transformer_decoder.layers[i]
#             attn_masks = [attn_mask, None]
#             query_feat = layer(
#                 query=query_feat,
#                 key=decoder_inputs[level_idx],
#                 value=decoder_inputs[level_idx],
#                 query_pos=query_embed,
#                 key_pos=decoder_positional_encodings[level_idx],
#                 attn_masks=attn_masks,
#                 query_key_padding_mask=None,
#                 key_padding_mask=None
#             )
#             orig_cls_pred, orig_mask_pred, attn_mask = self.forward_head(
#                 query_feat, mask_features,
#                 multi_scale_memorys[(i+1) % self.num_transformer_feat_level].shape[-2:], get_similarity=get_similarity
#             )
#             cls_pred = torch.cat([
#                 orig_cls_pred[:, :-(M+1), :],
#                 torch.mean(orig_cls_pred[:, -(M+1):, :], dim=1, keepdim=True)
#             ], dim=1)
#             if seg_mask is None:
#                 if self.mask_pred_method == 'max':
#                     mask_pred = orig_mask_pred
#                     cls_pred = orig_cls_pred
#                 else:
#                     mask_pred = orig_mask_pred
#                     cls_pred = orig_cls_pred
#             else:
#                 if self.mask_pred_method == 'max':
#                     mask_pred = torch.cat([
#                         orig_mask_pred[:, :-(M+1), :, :],
#                         torch.max(orig_mask_pred[:, -(M+1):, :, :], dim=1, keepdim=True)[0]
#                     ], dim=1)
#                     cls_pred = torch.cat([
#                         orig_cls_pred[:, :-(M+1), :],
#                         torch.mean(orig_cls_pred[:, -(M+1):, :], dim=1, keepdim=True)
#                     ], dim=1)

#                 else:
#                     mask_pred = orig_mask_pred
#                     cls_pred = orig_cls_pred
                    
#             cls_pred_list.append(cls_pred)
#             mask_pred_list.append(mask_pred)
        
#         if return_mask_features:
#             if self.gpt_bipartite_contrastive:
#                 return cls_pred_list, mask_pred_list, mask_features, ood_query_embeds
#             else:
#                 return cls_pred_list, mask_pred_list, mask_features
#         else:
#             if self.gpt_bipartite_contrastive:
#                 return cls_pred_list, mask_pred_list, ood_query_embeds
#             else:
#                 return cls_pred_list, mask_pred_list

#     def forward_train(self, x, texts, img_metas, gt_semantic_seg, train_cfg, gt_labels, gt_masks):
#         all_cls_scores, all_mask_preds, extra = self(x, texts, img_metas, seg_mask=gt_semantic_seg, return_mask_features=False)
#         # If bipartite contrastive mode is on, extra is ood_query_embeds.
#         if self.gpt_bipartite_contrastive:
#             ood_query_embeds = extra
#             losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks, img_metas, ood_query_embeds)
#         else:
#             losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks, img_metas)
#         return losses

#     def forward_test(self, inputs, texts, img_metas, test_cfg):
#         all_cls_scores, all_mask_preds = self(inputs, texts, img_metas)
#         cls_score, mask_pred = all_cls_scores[-1], all_mask_preds[-1]
#         ori_h, ori_w, _ = img_metas[0]['ori_shape']
#         if self.mask_pred_method == 'max':
#             M = self.M
#             mask_pred = torch.cat([
#                 mask_pred[:, :-(M+1), :, :],
#                 mask_pred[:, 21:22, :, :]  # 20 ~
#             ], dim=1)
#             cls_score = torch.cat([
#                 cls_score[:, :-(M+1), :],
#                 cls_score[:, 21:22, :]  # 20 ~
#             ], dim=1)
#             cls_score = F.softmax(cls_score, dim=-1)[..., :-1]
#             mask_pred = mask_pred.sigmoid()
#             seg_mask = torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred)
#         else:
#             cls_score = F.softmax(cls_score, dim=-1)[..., :-1]
#             mask_pred = mask_pred.sigmoid()
#             seg_mask = torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred)
#         return seg_mask

#     def forward_inference(self, inputs, texts, img_metas, test_cfg):
#         all_cls_scores, all_mask_preds, mask_features = self(inputs, texts, img_metas, return_mask_features=True)
#         return all_mask_preds, mask_features



# ################################################################################################################################


# import copy
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.cnn import Conv2d, build_plugin_layer, caffe2_xavier_init
# from mmcv.cnn.bricks.transformer import (build_positional_encoding,
#                                          build_transformer_layer_sequence)
# from mmcv.ops import point_sample
# from mmcv.runner import ModuleList, force_fp32
# from mmseg.models.builder import HEADS, build_loss
# from mmseg.models.decode_heads.decode_head import BaseDecodeHead

# from ...core import build_sampler, multi_apply, reduce_mean
# from ..builder import build_assigner
# from ..utils import get_uncertain_point_coords_with_randomness

# def mask_nms(mask_pred, cls_score, threshold=0.5, iou_threshold=0.5):
#     """
#     Non-Maximum Suppression (NMS) for masks
#     - threshold 값 이상을 가진 mask만 유지
#     - IOU가 높은 mask들에 대해 NMS 수행하여 중복 제거
#     """
#     batch_size, num_queries, h, w = mask_pred.shape
#     mask_pred = mask_pred.sigmoid()
#     # cls_prob = F.softmax(cls_score, dim=-1)  # (B, num_queries, num_classes+1)
#     cls_prob = cls_score
    
#     filtered_masks = []
#     for b in range(batch_size):
#         masks = mask_pred[b]  # (num_queries, H, W)
#         scores = cls_prob[b][:, :-1].max(dim=-1)[0]  # ID/OOD 구분 없이 가장 높은 클래스 확률

#         # Thresholding (최소 확률 이상만 유지)
#         keep = scores > threshold
#         masks = masks[keep]
#         scores = scores[keep]

#         # 중복된 mask 제거 (NMS)
#         keep_indices = []
#         for i in range(len(masks)):
#             if i in keep_indices:
#                 continue
#             keep_indices.append(i)
#             for j in range(i + 1, len(masks)):
#                 iou = (masks[i] * masks[j]).sum() / (masks[i] + masks[j] - masks[i] * masks[j]).sum()
#                 if iou > iou_threshold:
#                     if scores[j] > scores[i]:  # 신뢰도가 더 높은 mask만 남김
#                         keep_indices[-1] = j
        
#         filtered_masks.append(masks[keep_indices].sum(dim=0))  # Soft Merging 적용

#     return torch.stack(filtered_masks, dim=0)  # (B, H, W)

# # (아래 assigner 코드는 이미 별도 모듈에 등록되어 있으므로 build_assigner()를 통해 호출됨)
# # IdentityAssigner는 fixed matching용, MaskHungarianAssigner는 bipartite matching용 assigner이다.

# @HEADS.register_module()
# class tqdmHead(BaseDecodeHead):
#     """
#     Modified Mask2Former-style decoder head with separate ID and OOD query handling.

#     Query 분할:
#       - **ID Query:** indices 0 ~ (num_classes-1) (예: Cityscapes의 경우 0~18)
#       - **Fixed OOD Query:** index num_classes (즉, 19)
#       - **Learnable OOD Query:** indices (num_classes+1) ~ (num_queries-1)

#     Classification head의 출력은 총 (num_classes + 2) = 21채널 (ID: 0–18, OOD: 19, No-object: 20)이다.

#     _get_target_single()에서는 GT를 ID (label < num_classes)와 OOD (label == num_classes)로 분리한 후,
#     - ID query는 fixed matching(IdentityAssigner)을 사용하여 할당하고,
#     - OOD query는 bipartite matching(MaskHungarianAssigner)을 사용하여 매칭된 query는 label= num_classes (즉, 19, OOD)
#      로, 매칭되지 않은 query는 no-object (num_classes+1, 즉 20)로 할당한다.

#     추론 시에는 OOD query 영역에서 OOD class (채널 index = num_classes)의 확률이 가장 높은 query를 선택하여
#     해당 query의 mask 예측과 OOD 확률을 이용해 OOD segmentation을 산출한다.
#     """
#     def __init__(self,
#                  in_channels,
#                  feat_channels,
#                  out_channels,
#                  num_things_classes=80,
#                  num_stuff_classes=53,
#                  num_queries=100,
#                  num_transformer_feat_level=3,
#                  pixel_decoder=None,
#                  enforce_decoder_input_project=False,
#                  transformer_decoder=None,
#                  positional_encoding=None,
#                  loss_cls=None,
#                  loss_mask=None,
#                  loss_dice=None,
#                  train_cfg=None,
#                  test_cfg=None,
#                  init_cfg=None,
#                  text_proj=None,
#                  **kwargs):
#         # BaseDecodeHead 초기화 (input_transform은 multiple_select)
#         super(tqdmHead, self).__init__(
#             in_channels=in_channels,
#             channels=feat_channels,
#             # 여기서는 num_classes가 Cityscapes의 경우 19
#             num_classes=(num_things_classes + num_stuff_classes),
#             init_cfg=init_cfg,
#             input_transform='multiple_select',
#             **kwargs)

#         self.num_things_classes = num_things_classes
#         self.num_stuff_classes = num_stuff_classes
#         self.num_classes = self.num_things_classes + self.num_stuff_classes - 1  # 예: 19 (ID)
#         self.num_queries = num_queries  # 전체 query 수
#         self.num_transformer_feat_level = num_transformer_feat_level

#         self.num_heads = transformer_decoder.transformerlayers.attn_cfgs.num_heads
#         self.num_transformer_decoder_layers = transformer_decoder.num_layers

#         assert pixel_decoder.encoder.transformerlayers.attn_cfgs[0].num_levels == num_transformer_feat_level

#         pixel_decoder_ = copy.deepcopy(pixel_decoder)
#         pixel_decoder_.update(
#             in_channels=in_channels,
#             feat_channels=feat_channels,
#             out_channels=out_channels,
#             text_proj=text_proj)
#         self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]

#         self.transformer_decoder = build_transformer_layer_sequence(transformer_decoder)
#         self.decoder_embed_dims = self.transformer_decoder.embed_dims

#         self.decoder_input_projs = ModuleList()
#         for _ in range(num_transformer_feat_level):
#             if (self.decoder_embed_dims != feat_channels or enforce_decoder_input_project):
#                 self.decoder_input_projs.append(
#                     Conv2d(feat_channels, self.decoder_embed_dims, kernel_size=1))
#             else:
#                 self.decoder_input_projs.append(nn.Identity())

#         self.decoder_positional_encoding = build_positional_encoding(positional_encoding)

#         # Query embeddings
#         self.query_embed = nn.Embedding(self.num_queries, feat_channels)

#         self.text_proj = nn.Sequential(
#             nn.Linear(text_proj.text_in_dim, text_proj.text_out_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(text_proj.text_out_dim, text_proj.text_out_dim))

#         self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)

#         # ★ 수정: classification head 출력 차원을 (num_classes + 2)로 설정 (예: 19 + 2 = 21)
#         self.cls_embed = nn.Linear(feat_channels, self.num_classes + 2)
#         self.mask_embed = nn.Sequential(
#             nn.Linear(feat_channels, feat_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(feat_channels, feat_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(feat_channels, out_channels))

#         self.conv_seg = None

#         self.test_cfg = test_cfg
#         self.train_cfg = train_cfg

#         if train_cfg:
#             # fixed matching: ID query에 대해 IdentityAssigner 사용
#             self.id_assigner = self.ood_assigner = build_assigner(self.train_cfg.assigner)
#             # OOD query의 bipartite matching: MaskHungarianAssigner 사용
#             # 만약 train_cfg에 별도의 assigner_ood 설정이 있다면 사용하고, 없으면 기본 assigner config를 사용함
#             if hasattr(self.train_cfg, 'assigner_ood'):
#                 self.ood_assigner = build_assigner(self.train_cfg.assigner_ood)
#             else:
#                 self.ood_assigner = build_assigner(self.train_cfg.assigner)
#             self.sampler = build_sampler(self.train_cfg.sampler, context=self)
#             self.num_points = self.train_cfg.get('num_points', 12544)
#             self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
#             self.importance_sample_ratio = self.train_cfg.get('importance_sample_ratio', 0.75)
#         self.class_weight = loss_cls.class_weight
#         self.loss_cls = build_loss(loss_cls)
#         self.loss_mask = build_loss(loss_mask)
#         self.loss_dice = build_loss(loss_dice)
#         # 추가: OOD query 관련 loss weight (기본값 1.0)
#         self.loss_repulsion_weight = getattr(train_cfg, 'loss_repulsion_weight', 0.01) if train_cfg else 0.01
#         self.loss_bipartite_weight = getattr(train_cfg, 'loss_bipartite_weight', 1.0) if train_cfg else 1.0

#         # ★ 전체 query 구성:
#         # ID query: indices 0 ~ (num_classes-1)
#         # Fixed OOD query: index num_classes
#         # Learnable OOD query: indices (num_classes+1) ~ (num_queries-1)
#         self.M = self.num_queries - (self.num_classes + 1)

#     def init_weights(self):
#         for m in self.decoder_input_projs:
#             if isinstance(m, Conv2d):
#                 caffe2_xavier_init(m, bias=0)
#         self.pixel_decoder.init_weights()
#         for p in self.transformer_decoder.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_normal_(p)

#     def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
#                     gt_masks_list, img_metas):
#         (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
#          pos_inds_list, neg_inds_list) = multi_apply(
#             self._get_target_single, cls_scores_list,
#             mask_preds_list, gt_labels_list,
#             gt_masks_list, img_metas)
#         num_total_pos = sum((inds.numel() for inds in pos_inds_list))
#         num_total_neg = sum((inds.numel() for inds in neg_inds_list))
#         return (labels_list, label_weights_list, mask_targets_list,
#                 mask_weights_list, num_total_pos, num_total_neg)

#     def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks, img_metas):
#         """
#         수정된 target 할당:
#          - GT 분리: ID GT (label < num_classes)와 OOD GT (label == num_classes)
#          - ID query(0 ~ num_classes-1)는 IdentityAssigner를 사용하여 할당하고,
#          - OOD query (indices num_classes ~ end)는 MaskHungarianAssigner(즉, bipartite matching)를 사용하여
#            매칭된 경우 label = num_classes (즉, 19, OOD), 매칭되지 않은 경우 no-object (num_classes+1, 즉 20)로 할당한다.
#         """
#         num_queries = cls_score.shape[0]
#         id_num = self.num_classes  # 예: 19
#         ood_num = num_queries - id_num

#         # GT 분리: ID GT와 OOD GT
#         id_mask = gt_labels < self.num_classes
#         ood_mask = gt_labels == self.num_classes  # OOD GT는 label == self.num_classes (예: 19)
#         id_gt_labels = gt_labels[id_mask]
#         id_gt_masks = gt_masks[id_mask]
#         ood_gt_labels = gt_labels[ood_mask]
#         ood_gt_masks = gt_masks[ood_mask]

#         point_coords = torch.rand((1, self.num_points, 2), device=cls_score.device)

#         # ----- ID query 할당 (fixed matching) -----
#         if id_gt_labels.numel() > 0:
#             id_mask_pred = mask_pred[:id_num]  # (id_num, H, W)
#             mask_points_pred_id = point_sample(id_mask_pred.unsqueeze(1),
#                                                point_coords.repeat(id_num, 1, 1)).squeeze(1)
#             gt_points_masks_id = point_sample(id_gt_masks.unsqueeze(1).float(),
#                                               point_coords.repeat(id_gt_masks.shape[0], 1, 1)).squeeze(1)
#             assign_result_id = self.id_assigner.assign(cls_score[:id_num], mask_points_pred_id,
#                                                        id_gt_labels, gt_points_masks_id, img_metas)
#             sampling_result_id = self.sampler.sample(assign_result_id, id_mask_pred, id_gt_masks)
#             pos_inds_id = sampling_result_id.pos_inds
#             neg_inds_id = sampling_result_id.neg_inds
#             id_labels = id_gt_labels.new_full((id_num,), self.num_classes + 1, dtype=torch.long)
#             if len(pos_inds_id) > 0:
#                 id_labels[pos_inds_id] = id_gt_labels[sampling_result_id.pos_assigned_gt_inds]
#             id_label_weights = id_labels.new_ones(id_num)
#             id_mask_weights = id_mask_pred.new_zeros((id_num,))
#             id_mask_weights[pos_inds_id] = 1.0
#             id_mask_targets = id_gt_masks[sampling_result_id.pos_assigned_gt_inds] if len(pos_inds_id) > 0 else None
#         else:
#             id_labels = gt_labels.new_full((id_num,), self.num_classes + 1, dtype=torch.long)
#             id_label_weights = id_labels.new_ones(id_num)
#             id_mask_weights = mask_pred.new_zeros((id_num,))
#             pos_inds_id = torch.empty((0,), dtype=torch.long, device=cls_score.device)
#             neg_inds_id = torch.empty((0,), dtype=torch.long, device=cls_score.device)
#             id_mask_targets = None

#         # ----- OOD query 할당 (bipartite matching) -----
#         ood_pred = cls_score[id_num:]  # (ood_num, cls_out_channels)
#         if ood_gt_labels.numel() > 0:
#             ood_mask_pred = mask_pred[id_num:]
#             mask_points_pred_ood = point_sample(ood_mask_pred.unsqueeze(1),
#                                                 point_coords.repeat(ood_num, 1, 1)).squeeze(1)
#             gt_points_masks_ood = point_sample(ood_gt_masks.unsqueeze(1).float(),
#                                                point_coords.repeat(ood_gt_masks.shape[0], 1, 1)).squeeze(1)
#             assign_result_ood = self.ood_assigner.assign(ood_pred, mask_points_pred_ood,
#                                                          ood_gt_labels, gt_points_masks_ood, img_metas)
#             sampling_result_ood = self.sampler.sample(assign_result_ood, ood_mask_pred, ood_gt_masks)
#             pos_inds_ood = sampling_result_ood.pos_inds
#             neg_inds_ood = sampling_result_ood.neg_inds
#             ood_labels = ood_gt_labels.new_full((ood_num,), self.num_classes + 1, dtype=torch.long)
#             if len(pos_inds_ood) > 0:
#                 # 매칭된 OOD query는 label = self.num_classes (즉, 19, OOD)
#                 ood_labels[pos_inds_ood] = self.num_classes
#             ood_label_weights = ood_labels.new_ones(ood_num)
#             ood_mask_weights = ood_mask_pred.new_zeros((ood_num,))
#             ood_mask_weights[pos_inds_ood] = 1.0
#             ood_mask_targets = ood_gt_masks[sampling_result_ood.pos_assigned_gt_inds] if len(pos_inds_ood) > 0 else None
#         else:
#             ood_labels = gt_labels.new_full((ood_num,), self.num_classes + 1, dtype=torch.long)
#             ood_label_weights = ood_labels.new_ones(ood_num)
#             ood_mask_weights = mask_pred.new_zeros((ood_num,))
#             pos_inds_ood = torch.empty((0,), dtype=torch.long, device=cls_score.device)
#             neg_inds_ood = torch.empty((0,), dtype=torch.long, device=cls_score.device)
#             ood_mask_targets = None

#         labels = torch.cat([id_labels, ood_labels], dim=0)         # shape: (num_queries,)
#         label_weights = torch.cat([id_label_weights, ood_label_weights], dim=0)
#         mask_weights = torch.cat([id_mask_weights, ood_mask_weights], dim=0)
#         if (id_mask_targets is not None) and (ood_mask_targets is not None):
#             mask_targets = torch.cat([id_mask_targets, ood_mask_targets], dim=0)
#         elif id_mask_targets is not None:
#             mask_targets = id_mask_targets
#         elif ood_mask_targets is not None:
#             mask_targets = ood_mask_targets
#         else:
#             mask_targets = torch.empty((0, mask_pred.shape[-2], mask_pred.shape[-1]), device=mask_pred.device)

#         pos_inds = torch.cat([pos_inds_id, pos_inds_ood + id_num], dim=0)
#         neg_inds = torch.cat([neg_inds_id, neg_inds_ood + id_num], dim=0)

#         return (labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds)

#     def loss_single(self, cls_scores, mask_preds, gt_labels_list,
#                     gt_masks_list, img_metas):
#         """
#         Loss 계산:
#          - 기존 classification, mask, dice loss 외에
#          - OOD query 간 repulsion loss (feature 간 유사도 최소화)와 bipartite matching loss(placeholder)를 추가.
#         """
#         num_imgs = cls_scores.size(0)
#         cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
#         mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
#         (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
#          num_total_pos, num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list,
#                                                           gt_labels_list, gt_masks_list, img_metas)
#         labels = torch.stack(labels_list, dim=0)
#         label_weights = torch.stack(label_weights_list, dim=0)
#         mask_targets = torch.cat(mask_targets_list, dim=0) if len(mask_targets_list) > 0 else torch.empty((0, mask_preds.shape[-2], mask_preds.shape[-1]), device=mask_preds.device)
#         mask_weights = torch.stack(mask_weights_list, dim=0)

#         cls_scores = cls_scores.flatten(0, 1)
#         labels = labels.flatten(0, 1)
#         label_weights = label_weights.flatten(0, 1)
#         class_weight = cls_scores.new_tensor(self.class_weight)
#         loss_cls = self.loss_cls(
#             cls_scores,
#             labels,
#             label_weights,
#             avg_factor=class_weight[labels].sum())

#         num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
#         num_total_masks = max(num_total_masks, 1)

#         mask_preds_pos = mask_preds[mask_weights > 0]
#         if mask_targets.shape[0] == 0:
#             loss_dice = mask_preds_pos.sum()
#             loss_mask = mask_preds_pos.sum()
#         else:
#             with torch.no_grad():
#                 points_coords = get_uncertain_point_coords_with_randomness(
#                     mask_preds_pos.unsqueeze(1), None, self.num_points,
#                     self.oversample_ratio, self.importance_sample_ratio)
#                 mask_point_targets = point_sample(mask_targets.unsqueeze(1).float(),
#                                                   points_coords).squeeze(1)
#             mask_point_preds = point_sample(mask_preds_pos.unsqueeze(1), points_coords).squeeze(1)
#             loss_dice = self.loss_dice(
#                 mask_point_preds, mask_point_targets, avg_factor=num_total_masks)
#             mask_point_preds = mask_point_preds.reshape(-1, 1)
#             mask_point_targets = mask_point_targets.reshape(-1)
#             loss_mask = self.loss_mask(
#                 mask_point_preds,
#                 mask_point_targets,
#                 avg_factor=num_total_masks * self.num_points)

#         # OOD query 간 repulsion loss (쿼리 간 코사인 유사도 최소화)
#         id_num = self.num_classes
#         repulsion_loss = 0.0
#         bipartite_loss = 0.0  # placeholder; 실제로 matching cost 등을 이용하여 계산 가능
#         for i in range(num_imgs):
#             ood_cls = cls_scores_list[i][id_num:]
#             if ood_cls.shape[0] > 1:
#                 norm_feat = F.normalize(ood_cls, dim=-1)
#                 sim_matrix = norm_feat @ norm_feat.transpose(0, 1)
#                 repulsion_loss = repulsion_loss + (sim_matrix.sum() - torch.diag(sim_matrix).sum()) / (ood_cls.shape[0]*(ood_cls.shape[0]-1))
#         repulsion_loss = repulsion_loss / num_imgs

#         total_loss = loss_cls + loss_mask + loss_dice \
#                      + self.loss_repulsion_weight * repulsion_loss \
#                      + self.loss_bipartite_weight * bipartite_loss

#         return total_loss, loss_mask, loss_dice

#     @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
#     def loss(self, all_cls_scores, all_mask_preds, gt_labels_list, gt_masks_list, img_metas):
#         num_dec_layers = len(all_cls_scores)
#         all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
#         all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
#         img_metas_list = [img_metas for _ in range(num_dec_layers)]
#         losses = multi_apply(
#             self.loss_single, all_cls_scores, all_mask_preds,
#             all_gt_labels_list, all_gt_masks_list, img_metas_list)
#         losses_cls, losses_mask, losses_dice = losses
#         loss_dict = dict()
#         loss_dict['loss_cls'] = losses_cls[-1]
#         loss_dict['loss_mask'] = losses_mask[-1]
#         loss_dict['loss_dice'] = losses_dice[-1]
#         for num_dec_layer, (loss_cls_i, loss_mask_i, loss_dice_i) in enumerate(
#                 zip(losses_cls[:-1], losses_mask[:-1], losses_dice[:-1])):
#             loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
#             loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
#             loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
#         return loss_dict

#     def forward_head(self, decoder_out, mask_feature, attn_mask_target_size, get_similarity=False):
#         decoder_out = self.transformer_decoder.post_norm(decoder_out)
#         decoder_out = decoder_out.transpose(0, 1)  # (B, num_queries, c)
#         cls_pred = self.cls_embed(decoder_out)  # (B, num_queries, num_classes+2)
#         mask_embed = self.mask_embed(decoder_out)  # (B, num_queries, c)
#         mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
#         attn_mask = F.interpolate(mask_pred, attn_mask_target_size, mode='bilinear', align_corners=False)
#         attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat((1, self.num_heads, 1, 1)).flatten(0, 1)
#         attn_mask = (attn_mask.sigmoid() < 0.5).detach()
#         if not get_similarity:
#             return cls_pred, mask_pred, attn_mask
#         else:
#             mask_feature = F.normalize(mask_feature, dim=1, p=2)
#             mask_embed = F.normalize(mask_embed, dim=-1, p=2)
#             sim = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
#             sim = (sim + 1.) / 2.
#             return cls_pred, sim, attn_mask

#     def forward(self, feats, texts, img_metas, return_mask_features=False, get_similarity=False, seg_mask=None):
#         batch_size = len(img_metas)
#         mask_features, multi_scale_memorys = self.pixel_decoder(feats, texts)
#         decoder_inputs = []
#         decoder_positional_encodings = []
#         for i in range(self.num_transformer_feat_level):
#             decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
#             decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
#             level_embed = self.level_embed.weight[i].view(1, 1, -1)
#             decoder_input = decoder_input + level_embed
#             mask = decoder_input.new_zeros((batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool)
#             decoder_positional_encoding = self.decoder_positional_encoding(mask)
#             decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(2, 0, 1)
#             decoder_inputs.append(decoder_input)
#             decoder_positional_encodings.append(decoder_positional_encoding)

#         query_feat = self.text_proj(texts).permute(1, 0, 2)
#         query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))

#         cls_pred_list = []
#         mask_pred_list = []

#         # 초기 head 예측
#         orig_cls_pred, orig_mask_pred, attn_mask = self.forward_head(
#             query_feat, mask_features, multi_scale_memorys[0].shape[-2:], get_similarity=get_similarity)
#         cls_pred_list.append(orig_cls_pred)
#         mask_pred_list.append(orig_mask_pred)

#         for i in range(self.num_transformer_decoder_layers):
#             level_idx = i % self.num_transformer_feat_level
#             attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
#             layer = self.transformer_decoder.layers[i]
#             attn_masks = [attn_mask, None]
#             query_feat = layer(
#                 query=query_feat,
#                 key=decoder_inputs[level_idx],
#                 value=decoder_inputs[level_idx],
#                 query_pos=query_embed,
#                 key_pos=decoder_positional_encodings[level_idx],
#                 attn_masks=attn_masks,
#                 query_key_padding_mask=None,
#                 key_padding_mask=None
#             )
#             orig_cls_pred, orig_mask_pred, attn_mask = self.forward_head(
#                 query_feat, mask_features,
#                 multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:], get_similarity=get_similarity
#             )
#             cls_pred_list.append(orig_cls_pred)
#             mask_pred_list.append(orig_mask_pred)

#         if return_mask_features:
#             return cls_pred_list, mask_pred_list, mask_features
#         else:
#             return cls_pred_list, mask_pred_list

#     def forward_train(self, x, texts, img_metas, gt_semantic_seg, train_cfg,
#                       gt_labels, gt_masks):
#         all_cls_scores, all_mask_preds = self(x, texts, img_metas)
#         losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks, img_metas)
#         return losses
#     # # 방법 1 : max 사용
#     # def forward_test(self, inputs, texts, img_metas, test_cfg):
#     #     """
#     #     추론 시 OOD query 처리:
#     #       - 마지막 decoder layer의 출력에서,
#     #         * ID query 영역 (indices 0 ~ num_classes-1)는 기존 방식대로 처리하고,
#     #         * OOD query 영역 (indices num_classes ~ end)에서는 OOD class 확률(채널 index = num_classes)이 가장 높은 query를 선택하여
#     #           해당 query의 mask 예측과 OOD 확률을 사용해 OOD segmentation을 산출한다.
#     #       - 최종적으로 ID segmentation과 OOD segmentation을 결합하여 출력한다.
#     #     """
#     #     all_cls_scores, all_mask_preds = self(inputs, texts, img_metas)
#     #     cls_score, mask_pred = all_cls_scores[-1], all_mask_preds[-1]
#     #     ori_h, ori_w, _ = img_metas[0]['ori_shape']
#     #     batch_size = cls_score.shape[0]
#     #     cls_score = F.softmax(cls_score, dim=-1)

#     #     # ID query 처리 (indices 0 ~ num_classes-1)
#     #     # print("self.num_classes",self.num_classes)
#     #     id_cls_score = cls_score[:, :self.num_classes, :]
#     #     id_mask_pred = mask_pred[:, :self.num_classes, :, :]
#     #     # print("id_cls_score.shape",id_cls_score.shape)
#     #     # print("id_mask_pred.shape",id_mask_pred.shape)
#     #     # id_cls_prob = F.softmax(id_cls_score, dim=-1)
#     #     id_cls_prob = id_cls_score
#     #     # print("id_cls_prob.shape",id_cls_prob.shape)
#     #     id_mask_pred_sig = id_mask_pred.sigmoid()
#     #     seg_mask_id = torch.einsum('bqc,bqhw->bchw', id_cls_prob, id_mask_pred_sig)

#     #     # OOD query 처리 (indices self.num_classes ~ end)
#     #     ood_cls_score = cls_score[:, self.num_classes:, :]
#     #     ood_mask_pred = mask_pred[:, self.num_classes:, :, :]
#     #     # ood_probs = F.softmax(ood_cls_score, dim=-1)
#     #     ood_probs = ood_cls_score
#     #     # OOD class 확률은 전역 분포의 index self.num_classes (예: 19)
#     #     ood_prob_for_ood = ood_probs[:, :, self.num_classes]  # (B, ood_num)
#     #     best_ood_idx = ood_prob_for_ood.argmax(dim=1)  # (B,)
#     #     print(best_ood_idx)
#     #     print(ood_prob_for_ood)
#     #     # best_ood_cls = ood_cls_score[torch.arange(batch_size), best_ood_idx]
#     #     # best_ood_mask = ood_mask_pred[torch.arange(batch_size), best_ood_idx]
#     #     best_ood_cls = ood_cls_score[torch.arange(batch_size), best_ood_idx]
#     #     best_ood_mask = ood_mask_pred[torch.arange(batch_size), best_ood_idx]
#     #     best_ood_cls_prob = F.softmax(best_ood_cls, dim=-1)
#     #     ood_conf = best_ood_cls_prob[:, self.num_classes].unsqueeze(-1).unsqueeze(-1)
#     #     seg_mask_ood = best_ood_mask.sigmoid() * ood_conf
#     #     seg_mask_ood = seg_mask_ood.unsqueeze(1)  # (B, 1, H, W)

#     #     # 최종 segmentation: ID segmentation (self.num_classes 채널)와 OOD segmentation (1채널) 결합
#     #     seg_mask = torch.cat([seg_mask_id[:, :-2,:,:], seg_mask_ood], dim=1)
#     #     # seg_mask = F.interpolate(seg_mask, size=(ori_h, ori_w), mode='bilinear', align_corners=False)
#     #     return seg_mask
        
#     # 방법 2 - 1 : custom weighted sum
#     def forward_test(self, inputs, texts, img_metas, test_cfg):
#         """
#         추론 시 OOD query 처리 (후처리 개선 적용):
#           - ID query는 기존 방식대로 처리.
#           - OOD query는 가장 높은 확률을 가진 query 하나만 사용하는 대신,
#             모든 OOD query를 weighted sum 방식으로 병합하여 최종 OOD segmentation을 산출함.
#           - ID segmentation과 OOD segmentation을 결합하여 최종 출력.
#         """
#         all_cls_scores, all_mask_preds = self(inputs, texts, img_metas)
#         cls_score, mask_pred = all_cls_scores[-1], all_mask_preds[-1] # cls_score : [query, class]
#         ori_h, ori_w, _ = img_metas[0]['ori_shape']
#         batch_size = cls_score.shape[0]

#         cls_score = F.softmax(cls_score, dim=-1)
#         ### **Step 1: ID Query 처리** ###
#         id_cls_score = cls_score[:, :self.num_classes, :]
#         id_mask_pred = mask_pred[:, :self.num_classes, :, :]
#         # id_cls_prob = F.softmax(id_cls_score, dim=-1)
#         id_cls_prob = id_cls_score
#         id_mask_pred_sig = id_mask_pred.sigmoid()
#         seg_mask_id = torch.einsum('bqc,bqhw->bchw', id_cls_prob, id_mask_pred_sig)
    
#         ### **Step 2: OOD Query 처리 (Soft Merging 방식 적용)** ###
#         ood_cls_score = cls_score[:, self.num_classes:, :]
#         ood_mask_pred = mask_pred[:, self.num_classes:, :, :]
        
#         # OOD class 확률 계산 (softmax 적용)
#         # ood_probs = F.softmax(ood_cls_score, dim=-1)
#         ood_probs = ood_cls_score
#         # print("ood_probs.shape",ood_probs.shape)
#         # print("self.num_classes",self.num_classes)
#         ood_prob_for_ood = ood_probs[:, :, self.num_classes]  # (B, num_ood_queries)
#         # print("ood_prob_for_ood 19번째 cls의 확률들",ood_prob_for_ood)
#         # print("id_cls_score id의 확률들", id_cls_score)
        
#         # Soft merging 적용: 모든 OOD query의 가중치 합을 사용
#         ood_mask_pred_sig = ood_mask_pred.sigmoid()
        
#         weighted_ood_mask = torch.einsum('bq,bqhw->bqhw', ood_prob_for_ood, ood_mask_pred_sig)
#         temperature = 1
#         weighted_ood_mask = torch.sigmoid(temperature * (torch.max(ood_mask_pred_sig, dim=1)[0] - 0.5))
#         seg_mask_ood = weighted_ood_mask.unsqueeze(1)
        
#         # 차원 확장하여 batch-wise 처리
#         seg_mask_ood = weighted_ood_mask.unsqueeze(1)  # (B, 1, H, W)
    
#         ### **Step 3: 최종 Segmentation 생성** ###
#         # ID segmentation과 OOD segmentation 결합
#         seg_mask = torch.cat([seg_mask_id[:, :-2, :, :], seg_mask_ood], dim=1)
        
#         # 최종 크기로 보간
#         # seg_mask = F.interpolate(seg_mask, size=(ori_h, ori_w), mode='bilinear', align_corners=False)
        
#         return seg_mask
    
#     # # 방법 2 weighted sum
#     # def forward_test(self, inputs, texts, img_metas, test_cfg):
#     #     """
#     #     추론 시 OOD query 처리 (후처리 개선 적용):
#     #       - ID query는 기존 방식대로 처리.
#     #       - OOD query는 가장 높은 확률을 가진 query 하나만 사용하는 대신,
#     #         모든 OOD query를 weighted sum 방식으로 병합하여 최종 OOD segmentation을 산출함.
#     #       - ID segmentation과 OOD segmentation을 결합하여 최종 출력.
#     #     """
#     #     all_cls_scores, all_mask_preds = self(inputs, texts, img_metas)
#     #     cls_score, mask_pred = all_cls_scores[-1], all_mask_preds[-1]
#     #     ori_h, ori_w, _ = img_metas[0]['ori_shape']
#     #     batch_size = cls_score.shape[0]

#     #     cls_score = F.softmax(cls_score, dim=-1)
#     #     ### **Step 1: ID Query 처리** ###
#     #     id_cls_score = cls_score[:, :self.num_classes, :]
#     #     id_mask_pred = mask_pred[:, :self.num_classes, :, :]
#     #     # id_cls_prob = F.softmax(id_cls_score, dim=-1)
#     #     id_cls_prob = id_cls_score
#     #     id_mask_pred_sig = id_mask_pred.sigmoid()
#     #     seg_mask_id = torch.einsum('bqc,bqhw->bchw', id_cls_prob, id_mask_pred_sig)
    
#     #     ### **Step 2: OOD Query 처리 (Soft Merging 방식 적용)** ###
#     #     ood_cls_score = cls_score[:, self.num_classes:, :]
#     #     ood_mask_pred = mask_pred[:, self.num_classes:, :, :]
        
#     #     # OOD class 확률 계산 (softmax 적용)
#     #     ood_probs = F.softmax(ood_cls_score, dim=-1)
#     #     ood_probs = ood_cls_score
#     #     ood_prob_for_ood = ood_probs[:, :, self.num_classes]  # (B, num_ood_queries)
    
#     #     # Soft merging 적용: 모든 OOD query의 가중치 합을 사용
#     #     ood_mask_pred_sig = ood_mask_pred.sigmoid()
#     #     weighted_ood_mask = torch.einsum('bq,bqhw->bhw', ood_prob_for_ood, ood_mask_pred_sig)
        
#     #     # 차원 확장하여 batch-wise 처리
#     #     seg_mask_ood = weighted_ood_mask.unsqueeze(1)  # (B, 1, H, W)
    
#     #     ### **Step 3: 최종 Segmentation 생성** ###
#     #     # ID segmentation과 OOD segmentation 결합
#     #     seg_mask = torch.cat([seg_mask_id[:, :-2, :, :], seg_mask_ood], dim=1)
        
#     #     # 최종 크기로 보간
#     #     seg_mask = F.interpolate(seg_mask, size=(ori_h, ori_w), mode='bilinear', align_corners=False)
        
#     #     return seg_mask

#     # # 방법 3 : threshold        
#     # def forward_test(self, inputs, texts, img_metas, test_cfg):
#     #     """
#     #     추론 시 OOD query 처리 (Thresholding, NMS 및 Soft Merging 적용):
#     #       - ID segmentation과 OOD segmentation을 각각 후처리하여 최종 panoptic segmentation 생성
#     #     """
#     #     all_cls_scores, all_mask_preds = self(inputs, texts, img_metas)
#     #     cls_score, mask_pred = all_cls_scores[-1], all_mask_preds[-1]
#     #     ori_h, ori_w, _ = img_metas[0]['ori_shape']
#     #     batch_size = cls_score.shape[0]
    
#     #     ### **Step 1: ID Query 처리** ###
#     #     cls_score = F.softmax(cls_score, dim=-1)
#     #     id_cls_score = cls_score[:, :self.num_classes, :]
#     #     id_mask_pred = mask_pred[:, :self.num_classes, :, :]
#     #     # id_cls_prob = F.softmax(id_cls_score, dim=-1)
#     #     id_cls_prob = id_cls_score
#     #     id_mask_pred_sig = id_mask_pred.sigmoid()
#     #     seg_mask_id = torch.einsum('bqc,bqhw->bchw', id_cls_prob, id_mask_pred_sig)
    
#     #     ### **Step 2: OOD Query 처리 (Thresholding + NMS + Soft Merging)** ###
#     #     ood_cls_score = cls_score[:, self.num_classes:, :]
#     #     ood_mask_pred = mask_pred[:, self.num_classes:, :, :]
    
#     #     # Thresholding & NMS 적용
#     #     seg_mask_ood = mask_nms(mask_pred = ood_mask_pred, cls_score = ood_cls_score, threshold=0.5, iou_threshold=0.5)
        
#     #     # Batch 차원 추가 (B, 1, H, W)
#     #     seg_mask_ood = seg_mask_ood.unsqueeze(1)
    
#     #     ### **Step 3: 최종 Panoptic Segmentation 생성** ###
#     #     # ID segmentation과 OOD segmentation 결합
#     #     seg_mask = torch.cat([seg_mask_id[:, :-2, :, :], seg_mask_ood], dim=1)
        
#     #     # 최종 크기로 보간
#     #     seg_mask = F.interpolate(seg_mask, size=(ori_h, ori_w), mode='bilinear', align_corners=False)
        
#     #     return seg_mask
    
#     def forward_inference(self, inputs, texts, img_metas, test_cfg):
#         all_cls_scores, all_mask_preds, mask_features = \
#             self(inputs, texts, img_metas, return_mask_features=True)
#         return all_mask_preds, mask_features
