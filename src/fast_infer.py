"""
快速推理：绕开 mmdet/mmpose 的 test_step 后处理包装（predict_by_feat / predict），
只用模型前向（mode='tensor'）+ 手写紧凑后处理，显著降低每帧 Python 开销。

- fast_det_infer: 检测  RTMDet   cls_scores/bbox_preds → filter+topk → decode → NMS
- fast_pose_infer: 姿态  RTMPose  simcc → head.decode（原图坐标关键点）

前后处理与原高层 API 完全一致（复用 prior_generator / bbox_coder / head.decode），
只是去掉 predict 的 Python 包装层。
"""
import numpy as np
import torch
from mmcv.transforms import Compose as MMCVCompose
from mmcv.ops import nms as mmcv_nms
from mmengine.dataset import Compose as MMENGINECompose, pseudo_collate
from mmengine.registry import DefaultScope, init_default_scope
from mmdet.apis.inference import get_test_pipeline_cfg
from mmdet.models.utils.misc import filter_scores_and_topk


def make_fast_det_infer(detector):
    """快速检测推理闭包。
    infer(frame) -> (bboxes_xyxy_np (N,4), scores_np (N,)) 均在原图坐标系，已按分数降序
    """
    cfg = detector.cfg.copy()
    test_pipeline = get_test_pipeline_cfg(cfg)
    if test_pipeline[0].type == 'LoadImageFromFile':
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    with DefaultScope.overwrite_default_scope('mmdet'):
        test_pipeline = MMCVCompose(test_pipeline)

    preproc = detector.data_preprocessor
    head = detector.bbox_head
    # 推理侧后处理超参：单人监控场景不需要保留 3 万候选，收紧后 NMS 更快
    # score_thr=0.3 与 demo 的 s>0.3 过滤一致，nms_pre 从 30000 压到 1000
    nms_pre = 1000
    score_thr = 0.3
    max_per_img = 100
    iou_thr = 0.65

    def infer(frame):
        data_ = test_pipeline(dict(img=frame, img_id=0))
        meta = data_['data_samples'].metainfo
        # scale_factor: 处理图 → 原图 的缩放，用于还原坐标
        sf = meta.get('scale_factor', (1.0, 1.0))
        if isinstance(sf, (tuple, list)) and len(sf) == 2:
            sx, sy = float(sf[0]), float(sf[1])
        else:
            sx = sy = 1.0
        img_shape = meta['img_shape']           # 处理后的 (H, W)
        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]

        batch = preproc(data_, False)
        with torch.no_grad():
            cls_scores, bbox_preds = detector(batch['inputs'], mode='tensor')

        featmap_sizes = [s.shape[-2:] for s in cls_scores]
        # RTMDet 的回归距离已在 head 内部乘了 stride，因此 priors 只需 (x, y) 中心点
        mlvl_priors = head.prior_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype,
            device=cls_scores[0].device, with_stride=False)

        all_preds, all_priors, all_scores = [], [], []
        for cls_score, bbox_pred, priors in zip(cls_scores, bbox_preds, mlvl_priors):
            bbox_pred = bbox_pred[0].permute(1, 2, 0).reshape(-1, 4)
            cls_score = cls_score[0].permute(1, 2, 0).reshape(-1, head.cls_out_channels)
            scores = cls_score.sigmoid()
            f = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            all_scores.append(f[0])
            all_preds.append(f[3]['bbox_pred'])
            all_priors.append(f[3]['priors'])

        bbox_pred = torch.cat(all_preds)
        priors = torch.cat(all_priors)
        scores = torch.cat(all_scores)
        bboxes = head.bbox_coder.decode(priors, bbox_pred,
                                        max_shape=(img_shape[0], img_shape[1]))

        keep = mmcv_nms(bboxes, scores, iou_thr)[1]
        keep = keep[:max_per_img]
        bboxes, scores = bboxes[keep], scores[keep]

        # 还原到原图坐标
        bboxes = bboxes / torch.tensor([sx, sy, sx, sy], device=bboxes.device)
        # 截断到原图边界
        h0, w0 = frame.shape[:2]
        bboxes = bboxes.clamp(min=0)
        bboxes = torch.stack([
            bboxes[:, 0].clamp(max=w0 - 1), bboxes[:, 1].clamp(max=h0 - 1),
            bboxes[:, 2].clamp(max=w0 - 1), bboxes[:, 3].clamp(max=h0 - 1)], dim=1)
        # 分数降序
        order = scores.argsort(descending=True)
        return bboxes[order].cpu().numpy(), scores[order].cpu().numpy()

    return infer


def make_fast_pose_infer(pose_model):
    """快速姿态推理闭包（top-down, SimCC）。
    infer(frame, bboxes_xyxy) -> List[PoseDataSample]，keypoints 已在原图坐标
    """
    init_default_scope(pose_model.cfg.get('default_scope', 'mmpose'))
    pipeline = MMENGINECompose(pose_model.cfg.test_dataloader.dataset.pipeline)
    meta = pose_model.dataset_meta
    preproc = pose_model.data_preprocessor

    def infer(frame, bboxes):
        data_list = []
        for bbox in bboxes:
            data_info = dict(img=frame)
            data_info['bbox'] = bbox[None]
            data_info['bbox_score'] = np.ones(1, dtype=np.float32)
            data_info.update(meta)
            data_list.append(pipeline(data_info))
        if not data_list:
            return []
        batch = pseudo_collate(data_list)
        proc = preproc(batch, False)
        with torch.no_grad():
            simcc = pose_model(proc['inputs'], proc['data_samples'], mode='tensor')
            preds = pose_model.head.decode(simcc)

        # head.decode 返回的是输入图坐标，需按 bbox 的 scale/center 还原到原图
        # 公式: kpts_orig = (kpts / (input_size - 1)) * bbox_scales + (bbox_centers - bbox_scales/2)
        for p, ds in zip(preds, batch['data_samples']):
            gi = ds.gt_instances
            center = np.asarray(gi.bbox_centers, dtype=np.float32)   # (n,2)
            scale = np.asarray(gi.bbox_scales, dtype=np.float32)     # (n,2)
            inp = ds.metainfo['input_size']                          # (w, h) = (192, 256)
            # 与 mmpose 模型层 add_pred_to_datasample 完全一致:
            # keypoints / input_size * bbox_scales + bbox_centers - 0.5*bbox_scales
            denom = np.array([inp[0], inp[1]], dtype=np.float32)
            kpts = p.keypoints
            if isinstance(kpts, torch.Tensor):
                kpts = kpts.cpu().numpy()
            kpts = (kpts / denom) * scale + (center - scale / 2)
            p.keypoints = kpts.astype(np.float32)
        return preds

    return infer
