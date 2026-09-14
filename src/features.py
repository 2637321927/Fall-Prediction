"""
增强特征提取: 从 17 个关键点衍生更多几何特征
输入: (N, 17, 3) 关键点
输出: (N, feature_dim) 增强特征

可叠加特征:
  - 关节角度 (膝、髋、躯干倾斜...)
  - 关节速度 / 加速度
  - 重心高度 / 重心速度
  - 骨架高宽比
"""

import numpy as np


# 关键角度定义: (顶点, 一端, 另一端)
JOINT_ANGLES = [
    # 上半身
    ("left_elbow", 7, 5, 9),     # 左肘
    ("right_elbow", 8, 6, 10),    # 右肘
    ("left_shoulder", 5, 7, 11),  # 左肩
    ("right_shoulder", 6, 8, 12), # 右肩
    # 下半身
    ("left_hip", 11, 5, 13),      # 左髋
    ("right_hip", 12, 6, 14),     # 右髋
    ("left_knee", 13, 11, 15),    # 左膝
    ("right_knee", 14, 12, 16),   # 右膝
    # 躯干
    ("trunk_left", 5, 7, 11),     # 左躯干倾斜 (肩-肘-髋)
    ("trunk_right", 6, 8, 12),    # 右躯干倾斜
]


def joint_angle(kpts, apex, p1, p2):
    """计算关节角度 (p1-apex-p2)，kpts: (17, 3)"""
    v1 = kpts[p1, :2] - kpts[apex, :2]
    v2 = kpts[p2, :2] - kpts[apex, :2]
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.arccos(np.clip(cos, -1, 1))


def center_of_mass(kpts):
    """估算重心：髋关节中点"""
    lh, rh = kpts[11, :2], kpts[12, :2]
    if kpts[11, 2] < 0.3 or kpts[12, 2] < 0.3:
        return np.array([-1, -1])
    return (lh + rh) / 2


def extract_features(keypoints, window_size=16):
    """
    从关键点序列提取增强特征

    返回: (N, feature_dim) 特征矩阵
    feature_dim = 51 (原始) + 10 (角度) + 3 (重心) + 10 (速度) = 74
    """
    N = len(keypoints)
    features = []

    for i in range(N):
        feats = []

        # 1. 原始关键点 (51维)
        feats.append(keypoints[i].flatten())  # (51,)

        # 2. 关节角度 (10维)
        angles = []
        for _, apex, p1, p2 in JOINT_ANGLES:
            a = joint_angle(keypoints[i], apex, p1, p2)
            angles.append(a if not np.isnan(a) else 0)
        feats.append(np.array(angles))

        # 3. 重心特征 (3维): x, y, 高度
        com = center_of_mass(keypoints[i])
        head_y = keypoints[i, 0, 1]  # 鼻子 y
        ankle_y = (keypoints[i, 15, 1] + keypoints[i, 16, 1]) / 2
        feats.append(np.array([com[0], com[1], head_y - ankle_y]))  # 身高

        # 4. 帧间速度 (10维): 关键角度变化率
        if i > 0:
            vel = []
            for _, apex, p1, p2 in JOINT_ANGLES:
                a0 = joint_angle(keypoints[i-1], apex, p1, p2)
                a1 = joint_angle(keypoints[i], apex, p1, p2)
                vel.append(abs(a1 - a0) if not np.isnan(a1 - a0) else 0)
            feats.append(np.array(vel))
        else:
            feats.append(np.zeros(10))

        features.append(np.concatenate(feats))

    return np.array(features, dtype=np.float32)


# 增强后的特征维度
ENHANCED_FEATURE_DIM = 51 + 10 + 3 + 10  # = 74


def trunk_tilt(kpts):
    """躯干倾角: 肩中点-髋中点向量与竖直方向(上)的夹角(弧度). 0=直立, 近π/2=倒下"""
    ms = (kpts[5, :2] + kpts[6, :2]) / 2
    mh = (kpts[11, :2] + kpts[12, :2]) / 2
    v = ms - mh
    n = np.linalg.norm(v)
    if n < 1e-6:
        return 0.0
    cos = np.dot(v, np.array([0.0, -1.0])) / n
    return np.arccos(np.clip(cos, -1, 1))


def enhance_keypoints(keypoints):
    """
    为数据集准备增强特征 (方案B: 加入关节角, 提升视角鲁棒性)
    输入: (N, 17, 3) → 输出: (N, 17, 9)
    通道说明:
      0-2: 原始 x, y, conf
      3-4: 速度 vx, vy (帧间差分)
      5  : 关节相对重心角度
      6  : 到重心距离
      7  : 该关节的弯曲角(以它为顶点的关节角均值, 弧度; 非关节顶点为0)
      8  : 躯干倾角(广播到所有关节, 弧度) — 跌倒时躯干由直立转水平
    """
    N = keypoints.shape[0]
    enhanced = np.zeros((N, 17, 9), dtype=np.float32)
    enhanced[:, :, :3] = keypoints  # 原始 x, y, conf

    for i in range(N):
        com = center_of_mass(keypoints[i])
        tilt = trunk_tilt(keypoints[i])
        # 该帧每个关节作为顶点的角度(肩/髋各含2个角, 取均值)
        apex_angles = {}
        for _, apex, p1, p2 in JOINT_ANGLES:
            a = joint_angle(keypoints[i], apex, p1, p2)
            apex_angles.setdefault(apex, []).append(0.0 if np.isnan(a) else a)

        for j in range(17):
            enhanced[i, j, 8] = tilt  # 躯干倾角(广播到所有关节)
            if keypoints[i, j, 2] > 0.3:
                # 速度
                if i > 0 and keypoints[i-1, j, 2] > 0.3:
                    enhanced[i, j, 3:5] = keypoints[i, j, :2] - keypoints[i-1, j, :2]
                else:
                    enhanced[i, j, 3:5] = 0

                # 关节相对重心角度
                vec = keypoints[i, j, :2] - com
                enhanced[i, j, 5] = np.arctan2(vec[1], vec[0]) if com[0] > 0 else 0

                # 到重心距离
                enhanced[i, j, 6] = np.linalg.norm(vec) if com[0] > 0 else 0

                # 关节弯曲角 (方案B)
                if j in apex_angles:
                    enhanced[i, j, 7] = float(np.mean(apex_angles[j]))

    return enhanced


if __name__ == "__main__":
    kpts = np.random.randn(32, 17, 3).astype(np.float32)
    kpts[:, :, 2] = np.abs(kpts[:, :, 2])
    kpts[:, 11:13, 1] += 5  # 髋更低

    # 增强特征
    e = enhance_keypoints(kpts)
    print(f"增强特征: {kpts.shape} → {e.shape} (x,y,conf,vx,vy,angle,dist)")

    f = extract_features(kpts)
    print(f"提取特征: {f.shape} ({f.shape[1]}维)")
