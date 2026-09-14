"""
模块C: 状态机 - 基于连续 risk_score 的稳定预警策略
避免单帧误报，平衡召回率和精确率
"""

from collections import deque
from enum import Enum
import numpy as np


class AlertLevel(Enum):
    SAFE = "SAFE"                    # 安全
    LOW_RISK = "LOW_RISK"           # 低风险（关注）
    MID_RISK = "MID_RISK"           # 中风险
    HIGH_RISK = "HIGH_RISK"         # 高风险（预警）
    IMMEDIATE_ALERT = "IMMEDIATE"   # 摔倒/立即报警


class FallStateMachine:
    """
    跌倒预警状态机 - 基于 3 分类概率 [P0正常, P1不正常, P2摔倒] 的稳定预警

    3 分类概率规则:
    - P1(不正常) > 0.60 连续 10 窗口 → LOW_RISK
    - P1(不正常) > 0.75 连续 10 窗口 → MID_RISK
    - P1(不正常) > 0.90 连续 10 窗口 → HIGH_RISK
    - P2(摔倒)   > 0.90 连续 10 窗口 → IMMEDIATE_ALERT

    兼容旧: float(risk_score) 2分类 / int(0/1/2 类别)
    """

    def __init__(self, history_size: int = 15,
                 low_thresh: float = 0.60, low_count: int = 10,
                 mid_thresh: float = 0.75, mid_count: int = 10,
                 high_thresh: float = 0.90, high_count: int = 10,
                 imm_thresh: float = 0.90, imm_count: int = 10):
        """
        3 分类概率模式阈值/连续确认次数均可配置(update 传 probs=[P0,P1,P2]):
        - IMMEDIATE(摔倒): P2 > imm_thresh 连续 imm_count 次 (默认 10 次 >0.90)
        - HIGH_RISK     : P1 > high_thresh 连续 high_count 次 (默认 10 次 >0.90)
        - MID_RISK      : P1 > mid_thresh  连续 mid_count  次 (默认 10 次 >0.75)
        - LOW_RISK      : P1 > low_thresh  连续 low_count  次 (默认 10 次 >0.60)
        history_size 默认 15, 需 >= 各 count 才能完成连续确认判断
        """
        self.history: deque = deque(maxlen=history_size)
        self.current_level = AlertLevel.SAFE
        self.alert_count = 0
        self.low_thresh, self.low_count = low_thresh, low_count
        self.mid_thresh, self.mid_count = mid_thresh, mid_count
        self.high_thresh, self.high_count = high_thresh, high_count
        self.imm_thresh, self.imm_count = imm_thresh, imm_count

    def update(self, model_output) -> AlertLevel:
        """
        model_output: probs数组[P0,P1,P2] (3分类) / float(risk_score, 2分类) / int(0/1/2 类别)
        """
        self.history.append(model_output)
        h = list(self.history)

        if isinstance(model_output, (int, np.integer)):
            # 3分类类别模式: 0=normal, 1=falling, 2=fallen (兼容旧)
            if model_output == 2:
                self.current_level = AlertLevel.IMMEDIATE_ALERT
                self.alert_count += 1
                return self.current_level
            if len(h) >= 2 and all(x >= 1 for x in h[-2:]):
                self.current_level = AlertLevel.HIGH_RISK
                self.alert_count += 1
                return self.current_level
            if len(h) >= 3 and all(x >= 1 for x in h[-3:]):
                self.current_level = AlertLevel.LOW_RISK
                return self.current_level
            self.current_level = AlertLevel.SAFE
            return self.current_level

        if isinstance(model_output, (list, tuple, np.ndarray)) and np.asarray(model_output).size >= 3:
            # 3分类概率模式: [P0(正常), P1(不正常), P2(摔倒)]
            # 低->高渐进, 摔倒(P2)最高优先:
            #   P1>0.60 连续4 → LOW_RISK; P1>0.75 连续4 → MID_RISK;
            #   P1>0.90 连续4 → HIGH_RISK; P2>0.90 连续5 → IMMEDIATE_ALERT
            if len(h) >= self.imm_count and all(p[2] > self.imm_thresh for p in h[-self.imm_count:]):
                self.current_level = AlertLevel.IMMEDIATE_ALERT
                self.alert_count += 1
                return self.current_level
            if len(h) >= self.high_count and all(p[1] > self.high_thresh for p in h[-self.high_count:]):
                self.current_level = AlertLevel.HIGH_RISK
                self.alert_count += 1
                return self.current_level
            if len(h) >= self.mid_count and all(p[1] > self.mid_thresh for p in h[-self.mid_count:]):
                self.current_level = AlertLevel.MID_RISK
                return self.current_level
            if len(h) >= self.low_count and all(p[1] > self.low_thresh for p in h[-self.low_count:]):
                self.current_level = AlertLevel.LOW_RISK
                return self.current_level
            self.current_level = AlertLevel.SAFE
            return self.current_level

        # 2 分类模式（原始 risk_score）: 低->高渐进, 连续确认避免单帧误报
        if len(h) >= self.imm_count and all(x > self.imm_thresh for x in h[-self.imm_count:]):
            self.current_level = AlertLevel.IMMEDIATE_ALERT
            self.alert_count += 1
            return self.current_level
        if len(h) >= self.high_count and all(x > self.high_thresh for x in h[-self.high_count:]):
            self.current_level = AlertLevel.HIGH_RISK
            self.alert_count += 1
            return self.current_level
        if len(h) >= self.mid_count and all(x > self.mid_thresh for x in h[-self.mid_count:]):
            self.current_level = AlertLevel.MID_RISK
            return self.current_level
        if len(h) >= self.low_count and all(x > self.low_thresh for x in h[-self.low_count:]):
            self.current_level = AlertLevel.LOW_RISK
            return self.current_level
        self.current_level = AlertLevel.SAFE
        return self.current_level

    def reset(self):
        """重置状态"""
        self.history.clear()
        self.current_level = AlertLevel.SAFE
        self.alert_count = 0


def simulate_state_machine():
    """模拟测试状态机"""
    fsm = FallStateMachine()

    # 模拟一段风险逐渐升高的序列
    test_scores = [
        0.2, 0.3, 0.4, 0.5,     # 安全
        0.62, 0.65, 0.68,        # → LOW_RISK (连续3次>0.6)
        0.55, 0.45,              # → SAFE (回落)
        0.78, 0.82,              # → HIGH_RISK (连续2次>0.75)
        0.90,                    # → IMMEDIATE_ALERT
        0.30, 0.25               # → SAFE (恢复)
    ]

    print("=" * 50)
    print("状态机模拟测试")
    print("=" * 50)

    for i, score in enumerate(test_scores):
        level = fsm.update(score)
        bar = "█" * int(score * 20)
        print(f"帧{i:3d}  score={score:.2f} {bar:<20} → {level.value}")

    print(f"\n总预警次数: {fsm.alert_count}")


if __name__ == "__main__":
    simulate_state_machine()
