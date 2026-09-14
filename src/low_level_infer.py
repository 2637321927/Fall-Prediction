"""
低层直推优化：绕开 mmdet/mmpose 的高层 API（inference_detector / inference_topdown），
把 test_pipeline 只构建一次并复用，显著降低每帧 Python 开销。

与原高层 API 的前后处理完全一致（复刻官方实现），只是把"每帧重建 pipeline"移出循环。
"""
import numpy as np
import torch
from mmcv.transforms import Compose as MMCVCompose
from mmengine.dataset import Compose as MMENGINECompose, pseudo_collate
from mmengine.registry import DefaultScope, init_default_scope
from mmdet.apis.inference import get_test_pipeline_cfg


def make_det_infer(detector):
    """构建低层检测推理闭包。

    与 mmdet.apis.inference_detector 一致，但 test_pipeline 只构建一次。
    返回 infer(frame) -> DetDataSample（pred_instances 含 bboxes/scores）
    """
    cfg = detector.cfg.copy()
    test_pipeline = get_test_pipeline_cfg(cfg)
    # 输入始终是 numpy 帧，与 inference_detector 对 ndarray 的处理一致
    if test_pipeline[0].type == 'LoadImageFromFile':
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    # mmdet 的 transform（如 PackDetInputs）注册在 mmdet 下，构建时必须临时切回 mmdet 作用域
    # （否则会因当前默认作用域是 mmpose 而找不到，报 KeyError）
    with DefaultScope.overwrite_default_scope('mmdet'):
        test_pipeline = MMCVCompose(test_pipeline)

    def infer(frame):
        data_ = dict(img=frame, img_id=0)
        data_ = test_pipeline(data_)
        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]
        with torch.no_grad():
            return detector.test_step(data_)[0]

    return infer


def make_pose_infer(pose_model):
    """构建低层姿态推理闭包（top-down）。

    与 mmpose.apis.inference_topdown 一致，但 pipeline 只构建一次。
    返回 infer(frame, bboxes_xyxy) -> List[PoseDataSample]
    """
    init_default_scope(pose_model.cfg.get('default_scope', 'mmpose'))
    pipeline = MMENGINECompose(pose_model.cfg.test_dataloader.dataset.pipeline)
    meta = pose_model.dataset_meta

    def infer(frame, bboxes):
        # bboxes: (N, 4) xyxy
        data_list = []
        for bbox in bboxes:
            data_info = dict(img=frame)
            data_info['bbox'] = bbox[None]                     # (1, 4)
            data_info['bbox_score'] = np.ones(1, dtype=np.float32)
            data_info.update(meta)
            data_list.append(pipeline(data_info))
        if not data_list:
            return []
        batch = pseudo_collate(data_list)
        with torch.no_grad():
            return pose_model.test_step(batch)

    return infer
