# Ref-YOLO-World: RefYOLOWorldHeadModule / RefYOLOWorldHead
# CrossAttnContrastiveHead only. Forwards per_token_feats / per_token_mask.

import math
import copy
from typing import List, Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmengine.dist import get_dist_info
from mmengine.structures import InstanceData
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, InstanceList, OptInstanceList
from mmdet.models.utils import (multi_apply, unpack_gt_instances,
                                filter_scores_and_topk)
from mmyolo.registry import MODELS
from mmyolo.models.dense_heads import YOLOv8HeadModule, YOLOv8Head
from mmyolo.models.utils import gt_instances_preprocess
from mmcv.cnn.bricks import build_norm_layer

from ref_yolo_world.modules.ref_contrastive_blocks import CrossAttnContrastiveHead
from ref_yolo_world.modules.folded_ref_attn import fold_cross_attn_head


@MODELS.register_module()
class RefYOLOWorldHeadModule(YOLOv8HeadModule):
    """Head module for referring expressions.

    Uses CrossAttnContrastiveHead only and takes per_token_feats.
    """

    def __init__(self,
                 *args,
                 embed_dims: int,
                 use_einsum: bool = True,
                 freeze_all: bool = False,
                 use_positional_encoding: bool = False,
                 cross_attn_num_heads: int = 8,
                 fixed_scale_context: Optional[float] = None,
                 **kwargs) -> None:
        kwargs.pop('use_bn_head', None)
        self.embed_dims = embed_dims
        self.use_einsum = use_einsum
        self.freeze_all = freeze_all
        self.use_positional_encoding = use_positional_encoding
        self.cross_attn_num_heads = cross_attn_num_heads
        self.inference_fp32_modules: tuple[str, ...] = ()
        self.fixed_scale_context = fixed_scale_context
        super().__init__(*args, **kwargs)

    def set_folded_attn_params(
            self, txt_feats: Optional[Tensor],
            per_token_feats: Optional[Tensor],
            per_token_mask: Optional[Tensor] = None) -> None:
        if txt_feats is None or per_token_feats is None:
            for cls_contrast in self.cls_contrasts:
                cls_contrast.set_folded_attn_params(None)
            return
        reference_text = txt_feats[:, :1]
        for cls_contrast in self.cls_contrasts:
            cls_contrast.set_folded_attn_params(
                fold_cross_attn_head(
                    cls_contrast, per_token_feats, per_token_mask, reference_text))
    def set_inference_fp32_modules(
        self,
        fp32_modules: Optional[Sequence[str]],
    ) -> None:
        self.inference_fp32_modules = tuple(fp32_modules or ())

    def init_weights(self, prior_prob=0.01):
        super().init_weights()
        for cls_pred, cls_contrast, stride in zip(self.cls_preds,
                                                  self.cls_contrasts,
                                                  self.featmap_strides):
            cls_pred[-1].bias.data[:] = 0.0
            if hasattr(cls_contrast, 'bias'):
                nn.init.constant_(
                    cls_contrast.bias.data,
                    math.log(5 / self.num_classes / (640 / stride)**2))

    def _init_layers(self) -> None:
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.cls_contrasts = nn.ModuleList()

        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=reg_out_channels,
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=reg_out_channels,
                              out_channels=4 * self.reg_max,
                              kernel_size=1)))
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=self.embed_dims,
                              kernel_size=1)))
            self.cls_contrasts.append(
                CrossAttnContrastiveHead(
                    self.embed_dims,
                    self.norm_cfg,
                    num_heads=self.cross_attn_num_heads,
                    use_2d_positional_encoding=self.use_positional_encoding,
                    use_einsum=self.use_einsum,
                    fixed_scale_context=self.fixed_scale_context))

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)

        if self.freeze_all:
            self._freeze_all()

    def _freeze_all(self):
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()

    def forward(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
                txt_masks: Tensor,
                per_token_feats: Optional[Tensor] = None,
                per_token_mask: Optional[Tensor] = None) -> Tuple[List]:
        assert len(img_feats) == self.num_levels
        txt_feats_list = [txt_feats for _ in range(self.num_levels)]
        txt_masks_list = [txt_masks for _ in range(self.num_levels)]
        ptf_list = [per_token_feats for _ in range(self.num_levels)]
        ptm_list = [per_token_mask for _ in range(self.num_levels)]
        return multi_apply(self.forward_single, img_feats, txt_feats_list,
                           txt_masks_list, self.cls_preds, self.reg_preds,
                           self.cls_contrasts, ptf_list, ptm_list)

    def forward_single(self, img_feat: Tensor, txt_feat: Tensor,
                       txt_masks: Tensor, cls_pred: nn.ModuleList,
                       reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList,
                       per_token_feats: Optional[Tensor] = None,
                       per_token_mask: Optional[Tensor] = None) -> Tuple:
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed, txt_feat,
                                 fp32_modules=self.inference_fp32_modules,
                                 per_token_feats=per_token_feats,
                                 per_token_mask=per_token_mask)

        if txt_masks is not None:
            txt_masks = txt_masks.view(b, -1, 1, 1).expand(-1, -1, h, w)
            mask_fill_value = torch.finfo(cls_logit.dtype).min
            if self.training:
                cls_logit = cls_logit * txt_masks
            cls_logit = cls_logit.masked_fill(txt_masks == 0, mask_fill_value)

        bbox_dist_preds = reg_pred(img_feat)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds


@MODELS.register_module()
class RefYOLOWorldHead(YOLOv8Head):
    """YOLO-World Head for referring expressions.

    Forwards per_token_feats to head_module.
    loss / predict / predict_by_feat / loss_by_feat are the same as in YOLOWorldHead.
    """

    def __init__(self, world_size=-1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.world_size = world_size

    def set_folded_attn_params(
            self, txt_feats: Optional[Tensor],
            per_token_feats: Optional[Tensor],
            per_token_mask: Optional[Tensor] = None) -> None:
        if hasattr(self.head_module, "set_folded_attn_params"):
            self.head_module.set_folded_attn_params(
                txt_feats, per_token_feats, per_token_mask)
    def set_inference_fp32_modules(
        self,
        fp32_modules: Optional[Sequence[str]],
    ) -> None:
        if hasattr(self.head_module, 'set_inference_fp32_modules'):
            self.head_module.set_inference_fp32_modules(fp32_modules)

    def loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
             txt_masks: Tensor, batch_data_samples: Union[list, dict],
             per_token_feats: Optional[Tensor] = None,
             per_token_mask: Optional[Tensor] = None) -> dict:
        outs = self(img_feats, txt_feats, txt_masks,
                    per_token_feats=per_token_feats,
                    per_token_mask=per_token_mask)
        losses = self.loss_by_feat(
            *outs,
            batch_img_metas=batch_data_samples['img_metas'],
            batch_gt_instances=batch_data_samples['bboxes_labels'],
            batch_text_masks=txt_masks)
        return losses

    def loss_and_predict(
        self,
        img_feats: Tuple[Tensor],
        txt_feats: Tensor,
        txt_masks: Tensor,
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None,
        per_token_feats: Optional[Tensor] = None,
        per_token_mask: Optional[Tensor] = None
    ) -> Tuple[dict, InstanceList]:
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        outs = self(img_feats, txt_feats, txt_masks,
                    per_token_feats=per_token_feats,
                    per_token_mask=per_token_mask)

        loss_inputs = outs + (txt_masks, batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           cfg=proposal_cfg)
        return losses, predictions

    def forward(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
                txt_masks: Tensor,
                per_token_feats: Optional[Tensor] = None,
                per_token_mask: Optional[Tensor] = None) -> Tuple[List]:
        return self.head_module(img_feats, txt_feats, txt_masks,
                                per_token_feats=per_token_feats,
                                per_token_mask=per_token_mask)

    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                txt_masks: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False,
                per_token_feats: Optional[Tensor] = None,
                per_token_mask: Optional[Tensor] = None) -> InstanceList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(img_feats, txt_feats, txt_masks,
                    per_token_feats=per_token_feats,
                    per_token_mask=per_token_mask)
        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           rescale=rescale)
        return predictions

    def aug_test(self, aug_batch_feats, aug_batch_img_metas,
                 rescale=False, with_ori_nms=False, **kwargs):
        raise NotImplementedError('aug_test is not implemented yet.')

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_text_masks: Tensor,
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(
                mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_dist_preds = [
            bbox_pred_org.reshape(num_imgs, -1, self.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_dist_preds = torch.cat(flatten_dist_preds, dim=1)
        flatten_pred_bboxes = flatten_pred_bboxes * self.stride_tensor

        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(),
            self.flatten_priors_train,
            gt_labels,
            gt_bboxes,
            pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        assigned_scores = assigned_scores * batch_text_masks[:, :1, :]
        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos,
                assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            dist_mask = fg_mask_pre_prior.unsqueeze(-1).repeat(
                [1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(
                flatten_dist_preds, dist_mask).reshape([-1, 4, self.reg_max + 1])
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(
                pred_dist_pos.reshape(-1, self.reg_max + 1),
                assigned_ltrb_pos.reshape(-1),
                weight=bbox_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0

        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * num_imgs * world_size,
            loss_bbox=loss_bbox * num_imgs * world_size,
            loss_dfl=loss_dfl * num_imgs * world_size)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors, ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]

        results_list = []
        for (bboxes, scores, objectness,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_objectness, batch_img_metas):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            results = InstanceData(scores=scores,
                                   labels=labels,
                                   bboxes=bboxes[keep_idxs])

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                cfg.max_per_img = len(results)

            if results.bboxes.dtype != torch.float32:
                results.bboxes = results.bboxes.float()
            if results.scores.dtype != torch.float32:
                results.scores = results.scores.float()

            results = self._bbox_post_process(results=results,
                                              cfg=cfg,
                                              rescale=False,
                                              with_nms=with_nms,
                                              img_meta=img_meta)
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list
