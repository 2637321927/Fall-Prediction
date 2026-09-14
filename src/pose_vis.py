"""
骨架可视化工具: 在图像上绘制 17 个 COCO 关键点
"""

import cv2
import numpy as np

# COCO 骨架连线: (起点, 终点, BGR颜色)
EDGES = [
    (0, 1, (255, 200, 100)), (0, 2, (255, 200, 100)),
    (1, 3, (255, 200, 100)), (2, 4, (255, 200, 100)),
    (5, 6, (0, 255, 0)),
    (5, 7, (255, 0, 0)), (7, 9, (255, 0, 0)),
    (6, 8, (0, 0, 255)), (8, 10, (0, 0, 255)),
    (5, 11, (0, 255, 255)), (6, 12, (0, 255, 255)),
    (11, 12, (0, 255, 255)),
    (11, 13, (255, 255, 0)), (13, 15, (255, 255, 0)),
    (12, 14, (255, 0, 255)), (14, 16, (255, 0, 255)),
]

COLORS = [
    (255, 255, 255), (255, 200, 100), (255, 200, 100),
    (255, 200, 100), (255, 200, 100),
    (0, 255, 0), (0, 255, 0),
    (255, 0, 0), (0, 0, 255),
    (255, 0, 0), (0, 0, 255),
    (0, 255, 255), (0, 255, 255),
    (255, 255, 0), (255, 0, 255),
    (255, 255, 0), (255, 0, 255),
]


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray,
                  conf_threshold: float = 0.5) -> np.ndarray:
    """
    在 BGR 图像上绘制 17 点骨架
    keypoints: (17, 3) → [x, y, confidence]
    """
    h, w = frame.shape[:2]

    for p1, p2, color in EDGES:
        if keypoints[p1, 2] > conf_threshold and keypoints[p2, 2] > conf_threshold:
            x1, y1 = int(keypoints[p1, 0]), int(keypoints[p1, 1])
            x2, y2 = int(keypoints[p2, 0]), int(keypoints[p2, 1])
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    for i, (x, y, conf) in enumerate(keypoints):
        if conf > conf_threshold:
            px, py = int(x), int(y)
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(frame, (px, py), 5, COLORS[i], -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 7, (0, 0, 0), 1, cv2.LINE_AA)

    return frame
